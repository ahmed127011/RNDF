import resnet

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np

# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)
FLT_MAX = float(np.finfo(np.float32).max)


class FeatureLayer(nn.Sequential):
    def __init__(self, model_type='resnet34', num_output=256,
                 input_size=224, pretrained=False,
                 gray_scale=False):
        """
        Args:
            model_type (string): type of model to be used.
            num_output (int): number of neurons in the last feature layer
            input_size (int): input image size
            pretrained (boolean): whether to use a pre-trained model from ImageNet
            gray_scale (boolean): whether the input is gray scale image
        """
        super(FeatureLayer, self).__init__()
        self.model_type = model_type
        self.num_output = num_output
        if self.model_type == 'hybrid':
            # a model using a resnet-like backbone is used for feature extraction 
            model = resnet.Hybridmodel(self.num_output)
            self.add_module('hybrid_model', model)
        else:
            raise NotImplementedError

    def get_out_feature_size(self):
        return self.num_output


class Tree(nn.Module):
    def __init__(self, depth, feature_length, vector_length, use_cuda=True):
        """
        Args:
            depth (int): depth of the neural decision tree.
            feature_length (int): number of neurons in the last feature layer
            vector_length (int): length of the mean vector stored at each tree leaf node
            use_cuda (boolean): whether to use GPU
        """
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.feature_length = feature_length
        self.vector_length = vector_length
        self.is_cuda = use_cuda

        onehot = np.eye(feature_length)
        # randomly use some neurons in the feature layer to compute decision function
        using_idx = np.random.choice(feature_length, self.n_leaf, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        # a leaf node contains a mean vector and a covariance matrix
        self.mean = torch.ones((self.n_leaf, self.vector_length))
        # TODO: use k-means clusterring to perform leaf node initialization
        self.mu_cache = []
        # use sigmoid function as the decision function
        self.decision = nn.Sequential(OrderedDict([
            ('sigmoid', nn.Sigmoid()),
        ]))
        # used for leaf node update
        self.covmat = np.array([np.eye(self.vector_length) for i in range(self.n_leaf)])
        # also stores the inverse of the covariant matrix for efficiency
        self.covmat_inv = np.array([np.eye(self.vector_length) for i in range(self.n_leaf)])
        # also stores the determinant of the covariant matrix for efficiency
        self.factor = np.ones((self.n_leaf))
        if not use_cuda:
            raise NotImplementedError
        else:
            self.mean = Parameter(torch.from_numpy(self.mean).type(torch.FloatTensor).cuda(), requires_grad=False)
            self.covmat = Parameter(torch.from_numpy(self.covmat).type(torch.FloatTensor).cuda(), requires_grad=False)
            self.covmat_inv = Parameter(torch.from_numpy(self.covmat_inv).type(torch.FloatTensor).cuda(),
                                        requires_grad=False)
            self.factor = Parameter(torch.from_numpy(self.factor).type(torch.FloatTensor).cuda(), requires_grad=False)

    def forward(self, x, save_flag=False):
        """
        Args:
            param x (Tensor): input feature batch of size [batch_size, n_features]
        Return:
            (Tensor): routing probability of size [batch_size, n_leaf]
        """
        cache = {}
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()
        x = torch.mm(x, self.feature_mask)
        decision = self.decision(feats)

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)

        # save some intermediate results for analysis if needed
        if save_flag:
            cache['decision'] = decision[:, :, 0]
        batch_size = x.size()[0]

        mu = x.data.new(batch_size, 1, 1).fill_(1.)
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            # mu stores the probability that a sample is routed to certain node
            # repeat it to be multiplied for left and right routing
            mu = mu.repeat(1, 1, 2)
            # the routing probability at n_layer
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            mu = mu * _decision  # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)
            # merge left and right nodes to the same layer
            mu = mu.view(batch_size, -1, 1)
        mu = mu.view(batch_size, -1)

        if save_flag:
            cache['mu'] = mu
            return mu, cache
        else:
            return mu

    def pred(self, x):
        p = torch.mm(self(x), self.mean)
        return p

    def update_label_distribution(self, target_batch, check=False):
        """
        fix the feature extractor of RNDF and update leaf node mean vectors and covariance matrices
        based on a multivariate gaussian distribution
        Args:
            param target_batch (Tensor): a batch of regression targets of size [batch_size, vector_length]
        """
        target_batch = torch.cat(target_batch, dim=0)
        mu = torch.cat(self.mu_cache, dim=0)
        batch_size = len(mu)
        # no need for gradient computation
        with torch.no_grad():
            leaf_prob_density = mu.data.new(batch_size, self.n_leaf)
            for leaf_idx in range(self.n_leaf):
                # vectorized code is used for efficiency
                temp = target_batch - self.mean[leaf_idx, :]
                leaf_prob_density[:, leaf_idx] = (self.factor[leaf_idx] * torch.exp(
                    -0.5 * (torch.mm(temp, self.covmat_inv[leaf_idx, :, :]) * temp).sum(dim=1))).clamp(FLT_MIN,
                                                                                                       FLT_MAX)  # Tensor [batch_size, 1]
            nominator = (mu * leaf_prob_density).clamp(FLT_MIN, FLT_MAX)  # [batch_size, n_leaf]
            denomenator = (nominator.sum(dim=1).unsqueeze(1)).clamp(FLT_MIN, FLT_MAX)  # add dimension for broadcasting
            zeta = nominator / denomenator  # [batch_size, n_leaf]
            # new_mean if a weighted sum of all training samples
            new_mean = (torch.mm(target_batch.transpose(0, 1), zeta) / (zeta.sum(dim=0).unsqueeze(0))).transpose(0,
                                                                                                                 1)  # [n_leaf, vector_length]
            # allocate for new parameters
            new_covmat = new_mean.data.new(self.n_leaf, self.vector_length, self.vector_length)
            new_covmat_inv = new_mean.data.new(self.n_leaf, self.vector_length, self.vector_length)
            new_factor = new_mean.data.new(self.n_leaf)
            for leaf_idx in range(self.n_leaf):
                # new covariance matrix is a weighted sum of all covmats of each training sample
                weights = zeta[:, leaf_idx].unsqueeze(0)
                temp = target_batch - new_mean[leaf_idx, :]
                new_covmat[leaf_idx, :, :] = torch.mm(weights * (temp.transpose(0, 1)), temp) / (weights.sum())
                # update cache (factor and inverse) for future use
                new_covmat_inv[leaf_idx, :, :] = new_covmat[leaf_idx, :, :].inverse()
                if check and new_covmat[leaf_idx, :, :].det() <= 0:
                    print('Warning: singular matrix %d' % leaf_idx)
                new_factor[leaf_idx] = 1.0 / max((torch.sqrt(new_covmat[leaf_idx, :, :].det())), FLT_MIN)
        # update parameters
        self.mean = Parameter(new_mean, requires_grad=False)
        self.covmat = Parameter(new_covmat, requires_grad=False)
        self.covmat_inv = Parameter(new_covmat_inv, requires_grad=False)
        self.factor = Parameter(new_factor, requires_grad=False)
        return


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        feature_length = 30
        onehot = np.eye(feature_length)
        # randomly use some neurons in the feature layer to compute decision function
        using_idx = np.random.choice(feature_length, 10, replace=False)
        self.feature_mask = onehot[using_idx].T
        #self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        # a leaf node contains a mean vector and a covariance matrix

        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(12800,1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pred(self, x):
        #p = torch.mm(self(x), self.mean)
        return self(x)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()
        x = torch.mm(x, self.feature_mask)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
       # print(probas.shape)
       # print(logits.shape)
        return probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model

class Forest(nn.Module):
    # a neural decision forest is an ensemble of neural decision trees
    def __init__(self, n_tree, tree_depth, feature_length, vector_length, grayscale=True, use_cuda=False):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        self.tree_depth = tree_depth
        self.feature_length = feature_length
        self.vector_length = vector_length
        for _ in range(n_tree):
            tree = resnet34(feature_length, grayscale)
            self.trees.append(tree)

    def forward(self, x, save_flag=False):
        predictions = []
        cache = []
        for tree in self.trees:
            if save_flag:
                # record some intermediate results
                mu, cache_tree = tree(x, save_flag=True)
                p = torch.mm(mu, tree.mean)
                cache.append(cache_tree)
            else:
                p = tree.pred(x)
                #print(p.shape)
            predictions.append(p.unsqueeze(2))
        prediction = torch.cat(predictions, dim=2)
        prediction = torch.sum(prediction, dim=2) / self.n_tree
        if save_flag:
            return prediction, cache
        else:

            return prediction


class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x, debug=False, save_flag=False):

        feats, reg_loss = self.feature_layer(x)
       # print("input shape")
       # print(x.shape)
        if save_flag:
            # return some intermediate results
            pred, cache = self.forest(x, save_flag=True)
            return pred, reg_loss, cache
        else:

            pred = self.forest(x)

            return pred, reg_loss