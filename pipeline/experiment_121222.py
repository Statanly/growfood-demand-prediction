# from pathlib import Path
# import argparse
# import warnings
#
# warnings.filterwarnings("ignore")
#
# from loader import DatasetLoader
# from pipeline.transforms import (
#     MovingAvgTransform,
#     HoltWintersPredictTransform,
#     IsWeekendTransform,
#     WeekdayTransform
# )
# from loader.ts_dataset import TsDataset
#
#
# root = Path('../data/processed/')
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('product_name', type=str)
# parser.add_argument('city_id', type=int)
# parser.add_argument('until_date', type=str)
# args = parser.parse_args()
# # product_name = 'balance'
# # city_id = 1
# # until_date = '2022-12-20'
#
# loader = DatasetLoader(root, product_name=args.product_name, city_id=args.city_id, until_date=args.until_date)
# df = loader.get_dataset()
#
# # cut extra history. HoltWinters transform is really long
# last_year_date = '2020-11-03'
# df = df[df.date > last_year_date]
#
# trs = [
#     HoltWintersPredictTransform(name='hws_1d', column='target'),
#
#     WeekdayTransform(name='weekday', column='date'),
#     IsWeekendTransform(name='weekend', column='date'),
#
#     MovingAvgTransform(name='ma_7', column='target', value=7),
#
#     MovingAvgTransform(name='ma_14', column='target', value=14),
# ]
#
# for tr in trs:
#     df = tr.transform(df)
#
#
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
#
# from statistics import mean
# from sklearn.model_selection import train_test_split
# from tensorboardX import SummaryWriter
#
# from models.lstm_mlp import LstmMlp
#
#
# train, test = train_test_split(df, test_size=0.2, random_state=42)
#
# window_size = 7*3
# train = DataLoader(TsDataset(train, window_size), batch_size=100, collate_fn=TsDataset.collate_fn)
# test =  DataLoader(TsDataset(test, window_size), batch_size=100, collate_fn=TsDataset.collate_fn)
#
#
# loss = nn.L1Loss()
# model = LstmMlp(df.shape[1]-2)
#
# opt = torch.optim.Adadelta(model.parameters())
# writer = SummaryWriter(f'runs/exp_{args.product_name}_{args.city_id}_121222/')
#
# train_step = 0
# test_step = 0
#
#
# best_test_smape = 1
# best_test_epoch = None
#
# for epoch in range(150):
#     # if epoch % 10 == 0:
#     #     print(f'epoch {epoch}')
#
#     metrics = dict(loss=[], smape=[])
#
#     for x, y in train:
#         model.train()
#         opt.zero_grad()
#
#         pred = model(x)
#         err = loss(y, pred).mean()
#         err.backward()
#         opt.step()
#
#         err = err.item()
#         err_smape = ((pred - y).abs() / (pred.abs() + y.abs()) / 2).mean().item()
#
#         metrics['loss'].append(err)
#         metrics['smape'].append(err_smape)
#
#         writer.add_scalar('loss/step/train', err, train_step)
#         writer.add_scalar('smape/step/train', err_smape, train_step)
#
#     train_step += 1
#     writer.add_scalar('loss/epoch/train', mean(metrics['loss']), epoch)
#     writer.add_scalar('smape/epoch/train', mean(metrics['smape']), epoch)
#
#     metrics = dict(loss=[], smape=[])
#     for x, y in test:
#         model.eval()
#         with torch.no_grad():
#             pred = model(x)
#             err = loss(y, pred)
#
#             err = err.item()
#             err_smape = ((pred - y).abs() / (pred.abs() + y.abs()) / 2).mean().item()
#
#         metrics['loss'].append(err)
#         metrics['smape'].append(err_smape)
#
#         writer.add_scalar('loss/step/test', err, test_step)
#         writer.add_scalar('smape/step/test', err_smape, test_step)
#
#     test_step += 1
#     writer.add_scalar('loss/epoch/test', mean(metrics['loss']), epoch)
#     writer.add_scalar('smape/epoch/test', mean(metrics['smape']), epoch)
#
#     min_smape = min(metrics['smape'])
#     if min_smape < best_test_smape:
#         best_test_smape = min_smape
#         best_test_epoch = epoch
#
# print(f'achieve smape {best_test_smape:.3f} on epoch {best_test_epoch}')
# with open(f'res_261222/{args.product_name}_{args.city_id}_{args.until_date}.txt', 'w') as f:
#     f.write(str(best_test_smape))
#     f.write('\n')
#     f.write(str(best_test_epoch))
