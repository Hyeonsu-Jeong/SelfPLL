import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import pickle, os
import torchvision.models
from imutils import paths
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split

from scipy.cluster import hierarchy 

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}  # 라벨을 숫자로 매핑하는 딕셔너리

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        label = self.label_map[label]  # 라벨을 숫자로 매핑
        
        if self.transform:
            img = self.transform(img)
        return img, label
        

def get_feature(dataset, depth):
    
    feature_path = os.path.join('./feature/', dataset)
    os.makedirs(feature_path, exist_ok=True)
    train_feature_path = os.path.join(feature_path, 'train_feature.pkl')
    test_feature_path = os.path.join(feature_path, 'test_feature.pkl')
    
    train_label_path = os.path.join(feature_path, 'train_label.pkl')
    test_label_path = os.path.join(feature_path, 'test_label.pkl')
    
    if os.path.isfile(train_feature_path) and os.path.isfile(test_feature_path) and os.path.isfile(train_label_path) and os.path.isfile(test_label_path):
        with open(train_feature_path, 'rb') as f:
            train_feature = pickle.load(f)
        
        with open(test_feature_path, 'rb') as f:
            test_feature = pickle.load(f)
            
        with open(train_label_path, 'rb') as f:
            train_label = pickle.load(f)
        
        with open(test_label_path, 'rb') as f:
            test_label = pickle.load(f)
        
        return train_feature, train_label, test_feature, test_label
    
    if depth == 18:
        model = torchvision.models.resnet18(weights="DEFAULT")
            
    if depth == 34:
        model = torchvision.models.resnet34(weights="DEFAULT")
    
    if depth == 50:
        model = torchvision.models.resnet50(weights="DEFAULT")
        
    model.fc = nn.Identity()
    
    transform_default = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    
    transform_caltech = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    
    if dataset == 'caltech256':
        _ = datasets.Caltech256(root='./data', download=True)
        
        image_paths = list(paths.list_images('./data/caltech256/256_ObjectCategories'))

        data = []
        targets = []
        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]
            if "clutter" in label:
                continue
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            data.append(img)
            targets.append(label)
        
        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)
        
        trainset = CustomDataset(train_data, train_targets, transform_caltech)
        testset = CustomDataset(test_data, test_targets, transform_caltech)
        
    elif dataset == 'caltech101':
        _ = datasets.Caltech101(root='./data', download=True)
        transform_caltech = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        image_paths = list(paths.list_images('./data/caltech101/101_ObjectCategories'))

        data = []
        targets = []
        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]
            if label == "BACKGROUND_Google":
                continue
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            data.append(img)
            targets.append(label)
        
        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)
        
        trainset = CustomDataset(train_data, train_targets, transform_caltech)
        testset = CustomDataset(test_data, test_targets, transform_caltech)
        
    elif dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_default)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_default)
        
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_default)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_default)
    
    elif dataset == 'food101':
        trainset = datasets.Food101(root='./data', split='train', download=True, transform=transform_default)
        testset = datasets.Food101(root='./data', split='test', download=True, transform=transform_default)
    
    elif dataset == 'flowers102':
        trainset = datasets.Flowers102(root='./data', split='test', download=True, transform=transform_default)
        testset = datasets.Flowers102(root='./data', split='train', download=True, transform=transform_default)
    
    elif dataset == 'stanfordcars':
        trainset = datasets.StanfordCars(root='./data', split='train', download=True, transform=transform_default)
        testset = datasets.StanfordCars(root='./data', split='test', download=True, transform=transform_default)  
    
    else:
        raise ValueError("Invalid dataset!")
    
    
    trainloader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        
    model.eval()
    train_feature = None
    test_feature = None
    model.cuda()
    train_label = []
    test_label = []
    
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs = inputs.cuda()
        
        with torch.no_grad():
        # compute output
            outputs = model(inputs)
            if train_feature is None:
                train_feature = outputs.detach().cpu()
            
            else:
                train_feature = torch.concatenate((train_feature, outputs.detach().cpu()), dim=0)
        train_label.extend(targets.tolist())
    
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        inputs = inputs.cuda()
        
        with torch.no_grad():
        # compute output
            outputs = model(inputs)
            if test_feature is None:
                test_feature = outputs.detach().cpu()
            
            else:
                test_feature = torch.concatenate((test_feature, outputs.detach().cpu()), dim=0)
        test_label.extend(targets.tolist())
    
        
    with open(train_feature_path, 'wb') as f:
        pickle.dump(train_feature, f)
    
    with open(train_label_path, 'wb') as f:
        pickle.dump(train_label, f)
    
    with open(test_feature_path, 'wb') as f:
        pickle.dump(test_feature, f)
    
    with open(test_label_path, 'wb') as f:
        pickle.dump(test_label, f)
    
    return train_feature, train_label, test_feature, test_label

def get_confusable_class(dataset):
    
    feature_path = os.path.join('./feature', dataset.lower())
    with open(os.path.join(feature_path, 'ft_map_train.pkl'), 'rb') as f:
        ft_map = pickle.load(f)
    
    distance_matrix = 1 - ft_map['mean']
    
    Z = hierarchy.linkage(distance_matrix, method='complete')
    
    threshold = np.max(Z[:, 2]) * 0.8
    
    labels = hierarchy.fcluster(Z, threshold, criterion='distance')
    clusters = {}
    
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return labels

def get_corrupted_target(target, dataset, ctype, corruption):
    
    num_classes = len(list(set(target)))
    corr_matrix = np.zeros((num_classes, num_classes))
    if ctype == 'sym':
        corr_matrix += (corruption)/(num_classes-1)
        np.fill_diagonal(corr_matrix, 1 - corruption)
        
    elif ctype == 'asym':
        corr_matrix += (corruption)/(num_classes)
        np.fill_diagonal(corr_matrix, 1 - corruption)
        
        for idx in range(num_classes):
            corr_matrix[idx][(idx+1)%num_classes] += corruption/(num_classes)
    
    elif ctype == 'cc':
        conf_class_index = get_confusable_class(dataset)
        num_clusters = max(conf_class_index)
        
        count = [0] * (num_clusters + 1)
        
        for idx in conf_class_index:
            count[idx] += 1
        
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    corr_matrix[i][j] = 1 - corruption
                    
                elif conf_class_index[i] == conf_class_index[j]:
                    corr_matrix[i][j] = corruption / (count[conf_class_index[i]]-1)
    
    indices_list = [[] for _ in range(num_classes)]
    corrupted_target = [0]*len(target)
    
    for idx, label in enumerate(target):
        indices_list[label].append(idx)
    
    for label in range(num_classes):
        trainsition_prob = corr_matrix[label]
        num_samples = len(indices_list[label])
        corrupted = np.random.choice(num_classes, size=num_samples, p=trainsition_prob)
        
        for i in range(num_samples):
            corrupted_target[indices_list[label][i]] = corrupted[i]

    return corrupted_target


def partial_loss(output, target, q=0.7):
    p = F.softmax(output, dim=1)
    Yg = torch.gather(p, 1, target).ravel()
    Lq = ((1-(Yg)**q)/q)
    return torch.mean(Lq)
    
if __name__=='__main__':
    get_feature('caltech256', 34)