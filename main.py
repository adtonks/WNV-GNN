#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics


# In[5]:


# import data and create graphs
def load_data(k):
    df = pd.read_csv('/home/redacted/all_graph_k_'+str(k)+'_haversine.csv', engine='python')

    data_list = []

    # process by day
    for DAY, group in df.groupby('DAY'):
        NODELABEL = LabelEncoder().fit_transform(group.TRAPID)
        node_labels = group.loc[group.DAY==DAY,['NODELABEL','TRAPID']].sort_values('NODELABEL').TRAPID.drop_duplicates().values
        edges_to = [[int(x) for x in s.split(", ")] if type(s) is str else [] for s in group.EDGES]
        edges_from = []
        for i in range(len(edges_to)):
            # self-loop
            edges_to[i].insert(0, group.NODELABEL.values[i])
            edges_from.append([group.NODELABEL.values[i]]*len(edges_to[i]))
        edges_to = [item for sublist in edges_to for item in sublist]
        edges_from = [item for sublist in edges_from for item in sublist]

        x0 = torch.FloatTensor(group.DailyCoolingDegreeDays.values)
        x1 = torch.FloatTensor(group.DailyHeatingDegreeDays.values)
        x2 = torch.FloatTensor(group.DailyPrecipitation.values)
        x3 = torch.FloatTensor(group.WEEK_POS_0.values)

        x4 = torch.FloatTensor(group.DailyCoolingDegreeDays7.values)
        x5 = torch.FloatTensor(group.DailyHeatingDegreeDays7.values)
        x6 = torch.FloatTensor(group.DailyPrecipitation7.values)
        x7 = torch.FloatTensor(group.WEEK_POS_7.values)

        x8 = torch.FloatTensor(group.DailyCoolingDegreeDays14.values)
        x9 = torch.FloatTensor(group.DailyHeatingDegreeDays14.values)
        x10 = torch.FloatTensor(group.DailyPrecipitation14.values)
        x11 = torch.FloatTensor(group.WEEK_POS_14.values)

        x12 = torch.FloatTensor(group.DailyCoolingDegreeDays21.values)
        x13 = torch.FloatTensor(group.DailyHeatingDegreeDays21.values)
        x14 = torch.FloatTensor(group.DailyPrecipitation21.values)
        x15 = torch.FloatTensor(group.WEEK_POS_21.values)

        x16 = torch.FloatTensor(group.DailyCoolingDegreeDays28.values)
        x17 = torch.FloatTensor(group.DailyHeatingDegreeDays28.values)
        x18 = torch.FloatTensor(group.DailyPrecipitation28.values)
        x19 = torch.FloatTensor(group.WEEK_POS_28.values)

        x20 = torch.FloatTensor(group.DailyCoolingDegreeDays35.values)
        x21 = torch.FloatTensor(group.DailyHeatingDegreeDays35.values)
        x22 = torch.FloatTensor(group.DailyPrecipitation35.values)
        x23 = torch.FloatTensor(group.WEEK_POS_35.values)

        x24 = torch.FloatTensor(group.DailyCoolingDegreeDays42.values)
        x25 = torch.FloatTensor(group.DailyHeatingDegreeDays42.values)
        x26 = torch.FloatTensor(group.DailyPrecipitation42.values)
        x27 = torch.FloatTensor(group.WEEK_POS_42.values)

        x = torch.stack((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
                        x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27), dim=1)
        y = torch.FloatTensor(group.NEXTWKPOS.values)
        edge_index = torch.tensor([edges_from, edges_to], dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)
        data_list.append(data)

    # 2008 - 2016 (8.5 years)
    train_data_loader = DataLoader(data_list[0:1258], batch_size=100)
    # 2017 - 2018 (2 years)
    valid_data_loader = DataLoader(data_list[1258:1502], batch_size=100)
    # 2019 - 2021 (2.5 years)
    test_data_loader = DataLoader(data_list[1502:1764], batch_size=100)
    
    return train_data_loader, valid_data_loader, test_data_loader


# In[34]:


# define logistic, FC and GNN models
class Net(torch.nn.Module):
    def __init__(self, lag_wks, mod_type, vars_sel):
        super(Net, self).__init__()
        self.lag_wks = lag_wks
        if vars_sel=="all":
            self.cols = range(self.lag_wks*4-4, self.lag_wks*4)
        elif vars_sel=="trap-only":
            self.cols = range(self.lag_wks*4-1, self.lag_wks*4)
        elif vars_sel=="weather-only":
            self.cols = range(self.lag_wks*4-4, self.lag_wks*4-1)
        else:
            print("No such variable selection type")
        self.mod_type = mod_type
        if self.mod_type=="Logistic":
            self.lin1 = torch.nn.Linear(len(self.cols), 1)
        elif self.mod_type=="Fully-connected":
            self.lin1 = torch.nn.Linear(len(self.cols), 8)
            self.lin2 = torch.nn.Linear(8, 8)
            self.lin3 = torch.nn.Linear(8, 8)
            self.lin4 = torch.nn.Linear(8, 1)
        elif self.mod_type=="GNN":
            self.conv1 = SAGEConv(len(self.cols), 8)
            self.conv2 = SAGEConv(8, 8)
            self.conv3 = SAGEConv(8, 8)
            self.conv4 = SAGEConv(8, 1)
        else:
            print("No such model type")
    def forward(self, data):        
        x, edge_index = data.x[:, self.cols], data.edge_index
        if self.mod_type=="Logistic":
            x = self.lin1(x)
        elif self.mod_type=="Fully-connected":
            x = self.lin1(x)
            x = F.relu(x)
            x = self.lin2(x)
            x = F.relu(x)
            x = self.lin3(x)
            x = F.relu(x)
            x = self.lin4(x)
        elif self.mod_type=="GNN":
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = self.conv4(x, edge_index)
        return x


# In[7]:


device = torch.device('cuda')

def train(model, loader, crit, optimizer):
    loss_tot = 0.0
    dataset_len = 0
    for data in loader:
        data = data.to(device)
        log_odds_dev = torch.squeeze(model(data))
        loss = crit(log_odds_dev, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tot += loss.item()*len(data.y)
        dataset_len += len(data.y)
    return loss_tot/dataset_len


# In[8]:


def calc_loss(model, loader, crit):
    model.eval()
    loss_tot = 0.0
    dataset_len = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            log_odds_dev = torch.squeeze(model(data))
            loss = crit(log_odds_dev, data.y)
            loss_tot += loss.item()*len(data.y)
            dataset_len += len(data.y)
    return loss_tot/dataset_len


# In[9]:


# create loss plot using lists of loss values
def plot_loss(n_epochs, train_loss, valid_loss, title):
    plt.plot(list(range(n_epochs)), train_loss, color='black', label="Train loss")
    plt.plot(list(range(n_epochs)), valid_loss, color='black', linestyle='dashed', label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
#     plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


# In[10]:


# compute predicted probabilities and classes for pytorch models
def nn_comp_preds(model, loader):
    probabilities = []
    predictions = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            log_odds_dev = torch.squeeze(model(data))
            probs = torch.sigmoid(log_odds_dev.detach()).cpu().numpy()
            pred = np.int32(probs>0.5)
            label = np.int32(data.y.detach().cpu().numpy())
            probabilities.extend(probs)
            predictions.extend(pred)
            labels.extend(label)
    return labels, probabilities, predictions


# In[11]:


# calculate metrics using lists of true labels and predicted probabilities/classes
def calc_metrics(labels, probabilities, predictions):
    roc = metrics.roc_curve(labels, probabilities)
    plt.plot(roc[0], roc[1], color='black')
    plt.xlabel("False positive rate (1-specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.show()
    
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "Brier score", "AUC",
                    "True negatives", "False positives", "False negatives", "True positives"]]
    metric_vals = [metrics.accuracy_score(labels, predictions)] \
    + list(metrics.precision_recall_fscore_support(labels, predictions, pos_label=1, average="binary")[:3]) \
    + [metrics.brier_score_loss(labels, probabilities), metrics.roc_auc_score(labels, probabilities)] \
    + list(metrics.confusion_matrix(labels, predictions).flatten())
    metrics_dict = dict(zip(metric_names, metric_vals))
    
    return metrics_dict


# In[33]:


# from torchinfo import summary

# run one of the pytorch models at a given lag and return metrics
def run_nn_mod(lag_wks, mod_type, n_epochs, early_stop_epochs, k, vars_sel):
    model = Net(lag_wks, mod_type, vars_sel).to(device)
    crit = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    train_loss = []
    valid_loss = []
    for epoch in tqdm(range(n_epochs)):
        train_loss.append(train(model, train_data_loader, crit, optimizer))
        valid_loss.append(calc_loss(model, valid_data_loader, crit))
        # early stopping
        if epoch>=early_stop_epochs:
            loss_base = valid_loss[-(early_stop_epochs+1)]
            loss_comp = min(valid_loss[-early_stop_epochs:])
            # stop if no improvement in most recent early_stop_epochs
            if loss_base<loss_comp:
                break
    
    title = mod_type + " model loss (" + str(lag_wks) + " week forecast distance)"         if lag_wks==1 else         mod_type + " model loss (" + str(lag_wks) + " weeks forecast distance)"
    plot_loss(epoch+1, train_loss, valid_loss, title)
    
    labels, probabilities, predictions = nn_comp_preds(model, test_data_loader)
    metrics_dict = {"Model type":mod_type, "Forecast distance (weeks)":lag_wks, "k":k, "Features":vars_sel}
    metrics_dict.update(calc_metrics(labels, probabilities, predictions))
#     print(summary(model))
    return  metrics_dict, labels, probabilities


# In[13]:


# prepare data for use with xgboost function
from xgboost import XGBClassifier

xg_boost_cols = ["DailyCoolingDegreeDays", "DailyHeatingDegreeDays", "DailyPrecipitation", "WEEK_POS_0",
                 "DailyCoolingDegreeDays7", "DailyHeatingDegreeDays7", "DailyPrecipitation7", "WEEK_POS_7",
                 "DailyCoolingDegreeDays14", "DailyHeatingDegreeDays14", "DailyPrecipitation14", "WEEK_POS_14",
                 "DailyCoolingDegreeDays21", "DailyHeatingDegreeDays21", "DailyPrecipitation21", "WEEK_POS_21",
                 "DailyCoolingDegreeDays28", "DailyHeatingDegreeDays28", "DailyPrecipitation28", "WEEK_POS_28",
                 "DailyCoolingDegreeDays35", "DailyHeatingDegreeDays35", "DailyPrecipitation35", "WEEK_POS_35",
                 "DailyCoolingDegreeDays42", "DailyHeatingDegreeDays42", "DailyPrecipitation42", "WEEK_POS_42",
                 "NEXTWKPOS"]

df = pd.read_csv('/home/redacted/all_graph_k_1_haversine.csv', engine='python')
# 2008 - 2016 (8.5 years)
xg_boost_df_train = df.loc[0:54426, xg_boost_cols]
# 2017 - 2018 (2 years)
xg_boost_df_valid = df.loc[54426:67473, xg_boost_cols]
# 2019 - 2021 (2.5 years)
xg_boost_df_test = df.loc[67473:81481, xg_boost_cols]


# In[14]:


# run xgboost model with data at a given lag and return metrics
def run_xgboost_mod(lag_wks, k, vars_sel):
    if vars_sel=="all":
        cols = range(lag_wks*4-4, lag_wks*4)
    elif vars_sel=="trap-only":
        cols = range(lag_wks*4-1, lag_wks*4)
    elif vars_sel=="weather-only":
        cols = range(lag_wks*4-4, lag_wks*4-1)
    else:
        print("No such variable selection type")

    model = XGBClassifier(use_label_encoder=False)
    model.fit(xg_boost_df_train.iloc[:, cols], xg_boost_df_train.iloc[:, 28],
              eval_set=[(xg_boost_df_valid.iloc[:, cols], xg_boost_df_valid.iloc[:, 28])],
              early_stopping_rounds=10, eval_metric="logloss", verbose=0)
    
    labels = xg_boost_df_test.iloc[:, 28].tolist()
    probabilities = model.predict_proba(xg_boost_df_test.iloc[:, cols])[:, 1].tolist()
    predictions = [round(x) for x in probabilities]

    metrics_dict = {"Model type":"XGboost", "Forecast distance (weeks)":lag_wks, "k":k, "Features":vars_sel}
    metrics_dict.update(calc_metrics(labels, probabilities, predictions))
    
    return metrics_dict, labels, probabilities


# In[12]:


# loop to run the pytorch models at different lags
n_epochs = 1000
early_stop_epochs = 10
dict_lst = []
labels_lst = []
probs_lst = []


# In[13]:


k = 1
train_data_loader, valid_data_loader, test_data_loader = load_data(k)
for vars_sel in ["all", "trap-only", "weather-only"]:
    for mod_type in ["Logistic", "Fully-connected", "GNN"]:
        for lag_wks in range(1, 8):
            metrics_dict, labels, probabilities = run_nn_mod(lag_wks, mod_type, n_epochs, early_stop_epochs, k, vars_sel)
            dict_lst.append(metrics_dict)
            labels_lst.append(labels)
            probs_lst.append(probabilities)
    # loop to run xgboost model at various lags
    for lag_wks in range(1, 8):
        metrics_dict, labels, probabilities = run_xgboost_mod(lag_wks, k, vars_sel)
        dict_lst.append(metrics_dict)
        labels_lst.append(labels)
        probs_lst.append(probabilities)


# In[14]:


# run GNN for remaining value for k in kNN
for vars_sel in ["all", "trap-only", "weather-only"]:
    for k in [2, 5, 10, 20]:
        train_data_loader, valid_data_loader, test_data_loader = load_data(k)
        for lag_wks in range(1, 8):
            metrics_dict, labels, probabilities = run_nn_mod(lag_wks, "GNN", n_epochs, early_stop_epochs, k, vars_sel)
            dict_lst.append(metrics_dict)
            labels_lst.append(labels)
            probs_lst.append(probabilities)


# In[15]:


# create the dataframe of metrics and save it
results_df = pd.DataFrame(dict_lst)
results_df.to_csv("results_table_all.csv", index=False)
results_df


# In[17]:


# example of replotting ROC curve from stored labels and predicted probabilities
roc = metrics.roc_curve(labels_lst[0], probs_lst[0])
plt.plot(roc[0], roc[1])

plt.xlabel("False positive rate (1-specificity)")
plt.ylabel("True positive rate (sensitivity)")
plt.show()

