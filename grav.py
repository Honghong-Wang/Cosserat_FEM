import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    import scienceplots

    plt.style.use(['science'])
except ImportError as e:
    pass
x = np.array([0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
auto = np.array([4.014, 4.027, 4.2, 4.7, 19.3, 64.5, 1506])
fem = np.array([4.05, 4.06, 4.2, 4.7, 19.25, 64.5, 1550])

fig, ax = plt.subplots(1, 1)
ax.plot(x, auto, label="AUTO")
ax.plot(x, fem, label="FEM")
ax.legend()
plt.show(block=False)

fig2, axx = plt.subplots(1, 1, figsize=(9, 9))
max_load = 4.35
# max_load = 30 * E0 * i0
LOAD_INCREMENTS = 20  # Follower load usually needs more steps compared to dead or pure bending
x = -np.linspace(0, max_load, LOAD_INCREMENTS)
# l0 = -np.array([0.0, -0.059391331045526664, -0.1183614971646926, -0.17649331557885795, -0.23337752552008997, -0.2886166437982582, -0.3418286952236804, -0.3926507783670747, -0.4407424290196146, -0.48578874600668087, -0.5275032466975709, -0.5656304226083216, -0.5999479688794866, -0.6302686650859985, -0.6564418887581724, -0.6783547471160982, -0.6959328167950992, -0.7091404857182576, -0.7179808957013138, -0.7224954888045344])
# l005 = -np.array([0.0, -0.11820877830491118, -0.23307215856321906, -0.3413709302679578, -0.4401329942360741, -0.5267437398675281, -0.5990413276311498, -0.6553928994439445, -0.6947486170808775])
# l006 = -np.array([0.0, -0.10093294747962392, -0.19978436650889808, -0.29452995769625506, -0.3832581099108588, -0.4642218780857385, -0.5358858828904995, -0.5969667026953561, -0.6464655421284187])
# l007 = -np.array([0.0, -0.08605794040674186, -0.17082609570803212, -0.25304045112362916, -0.3314879533896572, -0.40503055746300526, -0.4726275909039443, -0.53335593832882, -0.5864276004465612])
# l008 = -np.array([0.0, -0.0735444506539265, -0.14628405177442927, -0.21742569600700543, -0.28619956766556504, -0.35187031021510234, -0.41374762911345253, -0.4711961572500201, -0.5236444231006545]
# )

df = pd.read_csv('assets/one.csv')
x = np.array([0.008, 0.007, 0.006, 0.005, 0.00])
marker_ = ['^', '.', '+', "x", "none"]
arr = df.to_numpy()
for i in range(len(x) - 1):
    axx.plot(df.columns, -arr[i, :], label=r"l = {cc}".format(cc=x[i]), marker=marker_[i])
axx.plot(df.columns, -arr[-1, :], label=r"Classical", linestyle="dashed")
axx.legend(fontsize=20)
axx.set_xlabel("Moment at free end", fontsize=25)
axx.set_ylabel("Transverse tip displacement", fontsize=25)
plt.show()


