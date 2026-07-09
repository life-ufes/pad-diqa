import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Trainer:
	def __init__(self, device, save_name, criterion, optimizer):
		self.device = device
		self.criterion = criterion
		self.save_name = save_name
		self.optimizer = optimizer

	def fit(self, model, max_epochs, train_loader, val_loader):
		
		model.to(self.device)

		print("\nIniciando Treinamento...")
		self.min_loss = float('inf')
		self.loss_history_eval, self.loss_history_train = [], []
		self.preds, self.scores, self.labels, self.imgs_id = None, None, None, None
		
		for epoch in range(max_epochs):
			train_loss = self._train_epoch(model, train_loader, self.optimizer, self.criterion, epoch+1, max_epochs)
			eval_validation = self.evaluate(model, val_loader)
			self.loss_history_eval.append(eval_validation['loss'])
			self.loss_history_train.append(train_loss)
			if eval_validation['loss'] < self.min_loss:
				self.min_loss = eval_validation['loss']
				self.preds = eval_validation['preds']
				self.scores = eval_validation['targets']
				self.labels = eval_validation['labels']
				self.imgs_id = eval_validation['ids']
				torch.save(model.state_dict(), f'{self.save_name}')
			print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {eval_validation['loss']:.3f}")

		model.load_state_dict(torch.load(self.save_name))

	def _train_epoch(self, model, iterator, optimizer, criterion, c_epoch, max_epochs):
		model.train()
		epoch_loss = 0
		
		with tqdm(total=len(iterator), ascii=True, desc='Epoch {}/{}: '.format(c_epoch, max_epochs), ncols=100) as t:
			for batch in iterator:
				output, score, label, img_id = self.get_logits_targets(model, batch)
				optimizer.zero_grad()
				output = output.squeeze(-1)
				loss = criterion(output, score.float())
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()

				t.update()
		return epoch_loss / len(iterator)

	def evaluate(self, model, iterator):
		model.to(self.device)
		model.eval()
		epoch_loss = 0
		all_preds, all_scores, all_labels, all_imgs = [], [], [], []
		with torch.no_grad():
			with tqdm(total=len(iterator), ascii=True, desc='Evaluating...: ', ncols=100) as t:
				for batch in iterator:
					output, score, label, img_id = self.get_logits_targets(model, batch)
					output = output.squeeze(-1)
					loss = self.criterion(output, score.float())
					epoch_loss += loss.item()
					preds = output
					all_preds.extend(preds.cpu().numpy())
					all_scores.extend(score.cpu().numpy())
					all_labels.extend(label.cpu().numpy())
					all_imgs.extend(img_id)
					t.update()
		all_preds = np.array(all_preds)
		all_scores = np.array(all_scores)
		all_labels = np.array(all_labels)

		return {
			'loss': epoch_loss / len(iterator),
			'preds': all_preds,
			'targets': all_scores,
			'labels': all_labels,
			'ids': all_imgs
		}

	def get_logits_targets(self, model, batch):
		x, score, label, img_id = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3]
		return model(x), score, label, img_id

