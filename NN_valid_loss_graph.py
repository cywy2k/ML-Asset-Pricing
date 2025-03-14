# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data extracted from your output
data = {
    (1e-05, 0.001): [
        0.04828164475556049, 0.044940168839982815, 0.04301225362966458,
        0.04224040793875853, 0.041870276961061687, 0.041573000657889575,
        0.04149337801047497, 0.04132157196808192, 0.041362988452116646,
        0.041283645059189036, 0.04143981325129668, 0.041517611851708755,
        0.04171594637187405, 0.04185526775351415, 0.04213323237167464
    ],
    (1e-05, 0.01): [
        0.03793861983447439, 0.032516874476439424, 0.03471226564660254,
        0.036511387456104986, 0.038734206347726285, 0.0412911047719212,
        0.04272012431950619
    ],
    (0.0001, 0.001): [
        0.04426913240200116, 0.04284107581608825, 0.041186851889102,
        0.039997977023530334, 0.03968109482795828, 0.040636579480229154,
        0.041630511818867594, 0.04279857977396912, 0.043404619157728225,
        0.04445864842273295
    ],
    (0.0001, 0.01): [
        0.033303058732094035, 0.034335830882709056, 0.031229048061908946,
        0.03026950884506934, 0.030404495127085183, 0.0324777178078269,
        0.03310056867647088, 0.03271157529929446, 0.03334461059421301
    ],
    (0.001, 0.001): [
        0.034553852708389364, 0.036615370177767344, 0.04406402906816867,
        0.040383492001435824, 0.03635893863004943, 0.03405854395694203,
        0.03250997534228696, 0.03142952248971495, 0.03046835599363678,
        0.02981606781637917, 0.029253661606667772, 0.028851061231560178,
        0.028478044680216245, 0.02799763169605285, 0.027640987135883834,
        0.027473821485829022, 0.027386813139956858, 0.027373256407574646,
        0.027346431458782818, 0.027291425516725414, 0.027193753165192902,
        0.02709971906410323, 0.027001990626255672, 0.026926324237138033,
        0.026930596886409655, 0.02697944077145722, 0.026933291538928945,
        0.02695225502571298, 0.026914505145719483, 0.02689758962434199,
        0.026943301665596664, 0.026982965853272214, 0.026946118192022875,
        0.026921836633442178, 0.026899920259084966
    ],
    (0.001, 0.01): [
        0.0372262226883322, 0.033918396957839526, 0.037713155021063156,
        0.038816352426591844, 0.03307494298658437, 0.03478569366658727,
        0.033468682732847005, 0.034140702388766736, 0.0335068352934387,
        0.03611533620601727
    ]
}

# Prepare data for plotting
lr = []
lambda1 = []
val_loss = []

for (lmbda, learning_rate), losses in data.items():
    for loss in losses:
        lr.append(learning_rate)
        lambda1.append(lmbda)
        val_loss.append(loss)

# Convert to numpy arrays
lr = np.array(lr)
lambda1 = np.array(lambda1)
val_loss = np.array(val_loss)

# Create a grid for the surface plot
lr_unique = np.unique(lr)
lambda1_unique = np.unique(lambda1)
X, Y = np.meshgrid(lr_unique, lambda1_unique)

# Prepare Z values for the surface plot
Z = np.full(X.shape, np.nan)

for (lmbda, learning_rate), losses in data.items():
    for i, loss in enumerate(losses):
        # Locate the position in the Z array
        x_index = np.where(lr_unique == learning_rate)[0][0]
        y_index = np.where(lambda1_unique == lmbda)[0][0]
        Z[y_index, x_index] = loss

# Create a 3D plot
fig = plt.figure(figsize=(12,8)) 
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Set labels
ax.set_xlabel('Learning Rate (lr)')
ax.set_ylabel('Lambda1 ($\lambda_1$)')
ax.set_zlabel('Validation Loss')

# Set title
ax.set_title('Validation Loss')

# Add color bar
cbar = plt.colorbar(surf)
cbar.set_label('Validation Loss')

# Show the plot
plt.show()



