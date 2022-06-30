import os
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from models.dgcnn import DGCNN
from models.SVCNN import ResNetModel, CM3DRModel
from tools.triplet_dataloader import CrossDataLoader
from tools.utils import calculate_accuracy
from center_loss import CrossModalCenterLoss

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.backends.cudnn.enabled = False

def training(args):
    if not os.path.exists("./checkpoints/ModelNet40"):
        os.makedirs("./checkpoints/ModelNet40")

    image_feature_extractor = ResNetModel()
    pointcloud_feature_extractor = DGCNN()

    model = CM3DRModel(image_feature_extractor, pointcloud_feature_extractor,num_classes=40, dim_bit = args.dim_bit)
    model.train(True)

    model = model.to('cuda')
    model = torch.nn.DataParallel(model)
    

    ce_loss_criterion = nn.CrossEntropyLoss()
    center_loss_criterion = CrossModalCenterLoss(num_classes=40, feat_dim=args.dim_bit, use_gpu=True)
    #mse loss
    mse_criterion = nn.MSELoss()
 
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer_centloss = optim.SGD(center_loss_criterion.parameters(), lr=0.001)

    train_set = CrossDataLoader(dataset = "ModelNet40", num_points = 1024, num_classes=40, dataset_dir="./dataset/",  partition='train')
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False,num_workers=0)

    iteration = 0

    for epoch in range(1000):
        for data in data_loader:
            pt, img, img_v, target, target_vec = data

            
            pt = pt.permute(0,2,1)
            target = target[:,0]

            img = img.to('cuda')
            img_v = img_v.to('cuda')
            pt = pt.to('cuda')
            target = target.to('cuda')
            target_vec = target_vec.to('cuda')

            optimizer.zero_grad()
            optimizer_centloss.zero_grad()

            img_pred, point_pred, img_feat, point_feat = model(pt, img, img_v)


            # ---------------------------------
            # compyte weight and laplace matrix
            W_Matrix = torch.zeros(img_pred.size()[0], img_pred.size()[0], dtype=torch.float32).to('cuda')
            for i in range(img_pred.size()[0]):
                for j in range(img_pred.size()[0]):
                    dis = torch.norm(img_feat[i] - img_feat[j], p=2, dim=0)
                    exp_dis = torch.exp(-dis)
                    W_Matrix[i, j] = exp_dis / (1.0)**2 + 0.000001

            # compute B
            A_feat = torch.sum(W_Matrix, dim=0)
            L_Matrix = torch.diag(A_feat) - W_Matrix + 0.000001

            B = torch.torch.linalg.inv(L_Matrix).mm(img_feat + point_feat)
            B = torch.tanh(B)
            # ---------------------------------

            ce_loss = ce_loss_criterion(point_pred, target) + ce_loss_criterion(img_pred, target)

            cmc_loss = center_loss_criterion(torch.cat((img_feat, point_feat), dim = 0), torch.cat((target, target), dim = 0))
            
            mse_loss = mse_criterion(torch.tanh(img_feat), B) + mse_criterion(torch.tanh(point_feat), B)

            g_loss = torch.norm(torch.sum(img_feat, dim = 0), p=2, dim=0) + torch.norm(torch.sum(point_feat, dim = 0), p=2, dim=0)
	
            loss = ce_loss + 0.001 * g_loss + 10.0 * cmc_loss +  0.1 * mse_loss 
            loss.backward()

            optimizer.step()

            for param in center_loss_criterion.parameters():
                param.grad.data *= (1. / 10.0)

            # update the parameters for the cmc_loss
            optimizer_centloss.step()

            img_acc = calculate_accuracy(img_pred, target)
            pt_acc = calculate_accuracy(point_pred, target)

            if iteration % 100 == 0:
                print('epoch: %d iter: %d  loss: %f' % (epoch, iteration, loss.item()))

            iteration = iteration + 1
            if((iteration+1) % 1000) ==0:
                print(' Save')
                with open("./checkpoints/ModelNet40" + str(iteration+1)+'-head_net.pkl', 'wb') as f:
                    torch.save(model, f)
                with open("./checkpoints/ModelNet40" + str(iteration+1)+'-image_feature_extractor.pkl', 'wb') as f:
                    torch.save(image_feature_extractor, f)
                with open("./checkpoints/ModelNet40" + str(iteration+1)+'-pointcloud_feature_extractor.pkl', 'wb') as f:
                    torch.save(pointcloud_feature_extractor, f)

            iteration = iteration + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int,  default=16)
    
    parser.add_argument('--dim_bit', type=int,  default=32)

    parser.add_argument('--dropout', type=float,  default=0.5)
    

    args = parser.parse_args()
    training(args)
