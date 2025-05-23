import numpy as np
from utils import AverageMeterSet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.distributions import Categorical
from tqdm import tqdm
import torch.nn.functional as F
import os
from torch.nn.utils import clip_grad_norm_
from modeling import BERT4ETH
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils import resample

class PyTorchAdamWeightDecayOptimizer(AdamW):
    """A basic Adam optimizer that includes L2 weight decay for PyTorch."""
    def __init__(self, params, learning_rate, weight_decay_rate=0.01,
                 beta1=0.9, beta2=0.999, epsilon=1e-6):
        """Constructs a AdamWeightDecayOptimizer for PyTorch."""
        super().__init__(params, lr=learning_rate, betas=(beta1, beta2),
                         eps=epsilon, weight_decay=weight_decay_rate)


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    # BCE loss: -[y*log(p) + (1-y)*log(1-p)]
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Convert logits to probabilities
    probas = torch.sigmoid(logits)
    
    # Compute pt (p_t)
    pt = probas * targets + (1 - probas) * (1 - targets)

    # Compute focal loss
    focal_weight = (1 - pt) ** gamma
    focal_loss = alpha * focal_weight * bce_loss

    return focal_loss.mean() 

class MLP(nn.Module):
    def __init__(self, args , model):
        super(MLP, self).__init__()
        self.dnn1 = nn.Linear(64, 128)
        self.activation = nn.GELU()
        self.dnn2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)  # 1 output node cho binary classification
        self.device = args.device
        self.model = model.to(self.device)

    def forward(self, model, input_ids, label, counts, values, io_flags, positions, is_training=True):
        mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)

        x = self.model.embedding(input_ids, counts, values, io_flags, positions)

        for transformer in self.model.transformer:
            x = transformer(x, mask)

        input_tensor = x[:, 0, :]
        X_balanced = x[:, 0, :]
        y_balanced = label

        batch_size , hidden = X_balanced.shape
        
        if is_training:
            label = list(label.view(-1))
            minority_data = []
            minority_class = []

            majority_data = []
            majority_class = []

            for i in range(len(label)):
                if label[i] == 1:
                    minority_data.append(input_tensor[i,:])
                    minority_class.append(1.0)
                else:
                    majority_data.append(input_tensor[i,:])
                    majority_class.append(0.0)
            minority_upsampled = resample(minority_data, 
                              replace=True,     # Cho phép thay thế
                              n_samples=len(majority_data),  # Tạo ra số lượng dữ liệu như lớp đa số
                              random_state=42)
   
            minority_upsampled_labels = np.array([1] * len(minority_upsampled))

            X_balanced = torch.stack(majority_data + minority_upsampled , dim = 0).to(self.device)
            y_balanced = torch.tensor(majority_class + list(minority_upsampled_labels)).to(self.device)
            perm = torch.randperm(X_balanced.size(0))
            X_balanced = X_balanced[perm]
            y_balanced = y_balanced[perm]


        out = self.activation(self.dnn1(X_balanced))
        output = self.activation(self.dnn2(out))
        logits = self.output(output).squeeze()
        y_hat =  torch.sigmoid(logits).squeeze()

        if is_training:
            loss = focal_loss_with_logits(logits,y_balanced)
            total_loss = loss
            return total_loss
        return y_hat

       
class BERT4ETHTrainer_finetune(nn.Module):
    def __init__(self, args, vocab, model , data_loader):
        super(BERT4ETHTrainer_finetune, self).__init__()
        self.args = args
        self.device = args.device
        self.vocab = vocab
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.num_epochs = args.num_epochs
        self.mlp = MLP(self.args, self.model).to(self.device)

        self.optimizer, self.lr_scheduler = self._create_optimizer()

    def calculate_loss(self, batch):
        address_id = batch[0]
        label = batch[1].squeeze()
        input_ids = batch[2]
        counts = batch[3]
        values = batch[4]
        io_flags = batch[5]
        positions = batch[6]
        input_mask = batch[7]

        total_loss = self.mlp(self.model,input_ids, label,counts, values, io_flags, positions) # B x T x V
        
        return total_loss 

    def train(self):
        assert self.args.ckpt_dir, "must specify the directory for storing checkpoint"
        accum_step = 0
        for epoch in range(self.num_epochs):
            # print("bias:", self.output_bias[:10])
            accum_step = self.train_one_epoch(epoch, accum_step)
            if (epoch+1) % 5 == 0 or epoch==0:
                self.save_model(epoch+1, self.args.ckpt_dir)
    
    def evaluate(self):
        self.model.eval()  # Chuyển mô hình về chế độ eval
        self.y_hat_list = []  # Khởi tạo danh sách lưu kết quả
        tqdm_dataloader = tqdm(self.data_loader)

        # Chạy qua toàn bộ dữ liệu
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm_dataloader):
                # Đưa batch vào đúng thiết bị
                batch = [x.to(self.device) for x in batch]
                address_id = batch[0]
                label = batch[1].squeeze()  # Lấy nhãn (đảm bảo loại bỏ chiều không cần thiết)
                input_ids = batch[2]
                counts = batch[3]
                values = batch[4]
                io_flags = batch[5]
                positions = batch[6]
                input_mask = batch[7]

                # Lấy output từ mô hình
                y_hat = self.mlp(self.model, input_ids, label, counts, values, io_flags, positions, is_training=False)  # B x T x V

                self.y_hat_list.append(y_hat)  

        tensor_list = [t.unsqueeze(0) for t in self.y_hat_list] 
        result = torch.cat(tensor_list, dim=1) 

        return result.squeeze().cpu().numpy()


    def load(self, ckpt_dir):
        state_dict = torch.load(ckpt_dir)
        # Bỏ các key không liên quan đến model
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss.")}
        self.model.load_state_dict(filtered_state_dict, strict=False)


    def infer_embedding(self):
        self.model.eval()
        tqdm_dataloader = tqdm(self.data_loader)
        embedding_list = []
        address_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
       
                address = batch[0]
                label = batch[1].squeeze()

                input_ids = batch[2]
                counts = batch[3]
                values = batch[4]
                io_flags = batch[5]
                positions = batch[6]

                h = self.model.embedding(input_ids, counts, values, io_flags, positions)  # B x T x V

                cls_embedding = h[:,0,:]
                embedding_list.append(cls_embedding)
                address_ids = address.squeeze().tolist()
                

                addresses = self.vocab.convert_ids_to_tokens(address_ids)
                address_list += addresses
                label_list += label.squeeze().tolist()

        return address_list,label_list


    def train_one_epoch(self, epoch, accum_step):
        self.model.train()

        tqdm_dataloader = tqdm(self.data_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):

            batch_size = batch[0].shape[0]
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 5.0)  # Clip gradients

            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            accum_step += 1
            tqdm_dataloader.set_description(
                'Epoch {}, Step {}, loss {:.6f} '.format(epoch+1, accum_step, loss.item()))

        return accum_step

    def save_model(self, epoch, ckpt_dir):
        print(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_dir = os.path.join(ckpt_dir, "epoch_" + str(epoch)) + ".pth"
        print("Saving model to:", ckpt_dir)
        torch.save(self.model.state_dict(), ckpt_dir)

    def _create_optimizer(self):
        """Creates an optimizer training operation for PyTorch."""
        num_train_steps = self.args.num_train_steps
        num_warmup_steps = self.args.num_warmup_steps
        for name, param in self.named_parameters():
            print(name, param.size(), param.dtype)

        optimizer = PyTorchAdamWeightDecayOptimizer([
            {"params": self.parameters()}
        ],
            learning_rate=self.args.lr,
            weight_decay_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-6
        )

        # Implement linear warmup and decay
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda step: min((step + 1) / num_warmup_steps, 1.0)
                                                      if step < num_warmup_steps else (num_train_steps - step) / (
                                                                  num_train_steps - num_warmup_steps))

        return optimizer, lr_scheduler
