import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd

# Early Stopping (Algorithm 6)
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience  
        self.counter = 0          
        self.best_score = float('inf')  
        self.best_params = None    
        self.early_stop = False    

    def step(self, val_loss, model):
        if val_loss < self.best_score:
            
            self.best_score = val_loss   
            self.best_params = model.state_dict()  
            self.counter = 0            
        else:
            self.counter += 1            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop



## Penalized Loss
class PenalizedLoss(nn.Module):
    def __init__(self):
        super(PenalizedLoss, self).__init__()

    def forward(self, output, target, model, lambda1):
        loss = nn.MSELoss()(output, target)
        
        # L1 regulization
        l1_regularization = sum(torch.norm(param, p=1) for param in model.parameters())
        loss += lambda1 * l1_regularization

        return loss


class NN(nn.Module):
    def __init__(self, layer_sizes):
        super(NN, self).__init__()
        layers = []
        input_size = layer_sizes[0]
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(input_size, layer_sizes[i]))
            layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(nn.ReLU())
            input_size = layer_sizes[i]
        layers.append(nn.Linear(layer_sizes[-1], 1))  
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

# self-define Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx].drop(labels=[self.target_col]).values  
        y = self.data.iloc[idx][self.target_col]  
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Training Loop using Adam and Early Stopping
def train_model_nn(model, train_loader, val_loader, num_epochs=100, lr=0.001, lambda1=1e-5, patience=5):
    criterion = PenalizedLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.float())   # Ensure data is in float format

            target = target.float().unsqueeze(1)  # Ensure target has shape [batch_size, 1]
            loss = criterion(output, target, model, lambda1)  # Pass lambda1 to the loss function
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0
        total_size = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data.float())

                target = target.float().unsqueeze(1)  # Ensure target has shape [batch_size, 1]
                val_loss += criterion(output, target, model, lambda1).item()* target.size(0) 
                total_size += target.size(0)

        val_loss /= total_size
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')

        if early_stopping.step(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(early_stopping.best_params)
            break

    return val_loss, model


def R_oos(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    return 1 - (np.dot((actual-predicted),(actual-predicted)))/(np.dot(actual,actual))

def Feature_importence(first_layer_weight, recursive_num):
    feature_names = df.columns.tolist()
    feature_names.remove("RET")

    first_layer_weight_sum=np.sum(np.abs(first_layer_weight),axis=0)

    df_ft = pd.DataFrame({
        "feature": feature_names,
        f"weight_{recursive_num}": first_layer_weight_sum
    })
    df_ft[f"importance_{recursive_num}"] = df_ft[f"weight_{recursive_num}"]/df_ft[f"weight_{recursive_num}"].sum()

    return df_ft.loc[:,["feature",f"importance_{recursive_num}" ]]






if __name__ =='__main__': 
    ## A =============================== IMPORT DATA ======================================
    df = pd.read_parquet("GKX_top20.parquet").set_index(["DATE","permno"])
    year = df.index.get_level_values(0).year.unique().tolist()


    ## B =============================== PARAMETER SETTING =================================
    input_size = df.shape[1] - 1  # num of features
    batch_size = 10000
    layer_configs = {1: [input_size, 32], 
            2: [input_size, 32, 16], 
            3: [input_size, 32, 16, 8], 
            4: [input_size, 32, 16, 8, 4],
            5: [input_size, 32, 16, 8, 4, 2]}

    # Define the parameter grid for λ1 (regularization) and LR (learning rate)
    param_grid = {'lambda1': [1e-5, 1e-4, 1e-3],  # Exploring values between 10^-5 and 10^-3
            'lr': [0.001, 0.01]             # Learning rate options
            }

    #results savage
    nn_models = {}
    predictions_dict ={}
    true_labels_dict ={}
    R_oos_dict = {}
    feat_import_dict = {}

    ## C ============================ TRAIN FOR 5 NN MODELS =====================================
    for j in range (1, 6):

        # build model
        #nn_models[f"NN{j}"] = NN(layer_configs[j])
        model_val = NN(layer_configs[j])
        print(f"Created NN{j} model with {len(layer_configs[j])-1} hidden layers")

        # save prediction results for all recursive training
        predictions_all = []
        true_labels_all = []
        feature_names = df.columns.tolist()
        feature_names.remove("RET")
        feature_names.append("sic2")
        feature_importance_all = pd.DataFrame({
            "feature": feature_names})

        # resursive training
        for i in range(int(0.3*len(year)),int(0.8*len(year))+1):
            print(i)
            ## 1. --------------DATA DIVISION----------------
            # divide data
            train_end_date = year[i]
            valid_start_date = year[i]
            valid_end_date = year[i]+int(0.2*len(year))
            test_start_date= valid_end_date
            test_end_date= test_start_date+1

            train_data = df.loc[(df.index.get_level_values(0).year < train_end_date)]
            valid_data = df.loc[(df.index.get_level_values(0).year >= valid_start_date) & (df.index.get_level_values(0).year<valid_end_date)]
            test_data = df.loc[(df.index.get_level_values(0).year >= test_start_date) & (df.index.get_level_values(0).year<test_end_date)]
            train_all_data = df.loc[(df.index.get_level_values(0).year < valid_end_date)]  #train+valid data

            # load dataset
            train_dataset = TimeSeriesDataset(train_data, target_col='RET')
            valid_dataset = TimeSeriesDataset(valid_data, target_col='RET')
            test_dataset = TimeSeriesDataset(test_data, target_col='RET')
            train_all_dataset = TimeSeriesDataset(train_all_data, target_col='RET')

            # create DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            train_all_loader = DataLoader(train_all_dataset, batch_size=batch_size, shuffle=False)

            ## 2. ---------------PARAMETER TUNING---------------------
            # save best model, params, loss for each recursion
            best_loss = float('inf')
            best_params = None
            best_model = None

            for params in ParameterGrid(param_grid):
                print(params)
                # train model 
                val_loss,tra_model = train_model_nn(model_val, train_loader, valid_loader, lr=params['lr'], lambda1=params['lambda1'])

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    #best_model = nn_models[f"NN{j}"]  # save model status

            
            print(f"Best parameters: λ1={best_params['lambda1']}, LR={best_params['lr']}")
            print(f"Best validation loss: {best_loss}")
            
            # train the model with best para using train and valid sets
            new_model = NN(layer_configs[j])
            loss, best_model = train_model_nn(new_model, train_all_loader, valid_loader, lr=best_params['lr'], lambda1=best_params['lambda1'])

            # save best model
            model_filename = f'model_nn{j}_recursion_{i}.pth'  # actively generate file name
            torch.save(best_model.state_dict(), model_filename)
            nn_models[f"NN{j}"]=best_model 


            ## 3. ------------------- FEATURE IMPORTANCE -----------------------
            first_layer_weight = best_model.model[0].weight.detach().numpy()

            feature_import = Feature_importence(first_layer_weight,recursive_num=i)  #feature importance df
            feature_importance_all = pd.merge(feature_importance_all,feature_import,on="feature")


            ## 4. ---------------TEST DATA PREDICTION AND VALUATION METRICES------------------

            # predict test_loader
            best_model.eval()  # set into evaluation model

            # save prediction results for each recursive training
            predictions = []
            true_labels = []

            with torch.no_grad():  
                for data in test_loader:
                    inputs, labels = data
                    outputs = best_model(inputs)  # predict
                    predictions.extend(outputs.numpy())  
                    true_labels.extend(labels.numpy())  
            predictions_all.extend(predictions)
            true_labels_all.extend(true_labels)

        predictions_dict[f'model_{j}'] = predictions_all
        true_labels_dict[f'model_{j}'] = true_labels_all
        R_oos_dict[f'model_{j}'] = R_oos(true_labels_all, predictions_all)
        feat_import_dict[f'model_{j}'] = feature_importance_all

    

    ## D =============================== SAVE FINAL RESULTS IN PARQUET ===================================
    for j in range (1, 6):
        print(j)
        # prediction results
        pred_test_df = pd.DataFrame(predictions_dict[f'model_{j}'],columns=[f"pred_model_{j}"])
        true_test_df = pd.DataFrame(true_labels_dict[f'model_{j}'],columns=[f"true_model_{j}" ])
        pred_true_test_df = pd.concat([pred_test_df,true_test_df],axis=1)
        print(pred_true_test_df)
        pred_true_test_df.to_parquet(f"pred_results_with_true_model_{j}.parquet")

        #feature importance
        print(feat_import_dict[f'model_{j}'])
        feat_import_dict[f'model_{j}'].to_parquet(f"feature_import_model_{j}.parquet")

    #R_oos
    R_oos_df = pd.DataFrame(R_oos_dict, index=[0])
    print(R_oos_df)
    R_oos_df.to_parquet("Roos_results.parquet")