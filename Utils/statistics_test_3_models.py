import pandas as pd
from scipy.stats import shapiro

alpha = 0.05
print('Alpha = ',alpha)

print('ADNI')

# read csv

csv_path = 'output_adni.csv'
data_df = pd.read_csv(csv_path, header=None)

# store data in list
model1 = data_df.loc[0,:].tolist()
model2 = data_df.loc[1,:].tolist()
model3 = data_df.loc[2,:].tolist()

# convert data to float

model1 = [float(x) for x in model1]
model2 = [float(x) for x in model2]
model3 = [float(x) for x in model3]

# boxplot

# import matplotlib.pyplot as plt

#plt.figure()
#plt.boxplot([model1,model2,model3])
#plt.show()

# normality test model1

stat, p1 = shapiro(model1)
print('Statistics=%.3f, p=%.3f' % (stat, p1))
# interpret
if p1 > alpha:
	print('Shapiro model1: Model1 looks Gaussian (fail to reject H0)',p1)
else:
	print('Shapiro model1: Model1 does not look Gaussian (reject H0)',p1)

stat, p2 = shapiro(model2)
print('Statistics=%.3f, p=%.3f' % (stat, p2))
# interpret
if p2 > alpha:
	print('Shapiro model2: Model2 looks Gaussian (fail to reject H0)',p2)
else:
	print('Shapiro model2: Model2 does not look Gaussian (reject H0)',p2)

stat, p3 = shapiro(model3)
print('Shapiro: Statistics=%.3f, p=%.3f' % (stat, p3))
# interpret
if p3 > alpha:
	print('Shapiro model3: Model3 looks Gaussian (fail to reject H0)',p3)
else:
	print('Shapiro model3: Model3 does not look Gaussian (reject H0)',p3)

# significancy test
if p1 > alpha and p2 > alpha and p3 > alpha:

	print('All have normal distribution, using parametric method ANOVA')
	from scipy.stats import f_oneway
	stat, p_anova = f_oneway(model1, model2, model3)
	print('Anova: Statistics=%.3f, p=%.3f' % (stat, p_anova))
	# interpret
	if p_anova > alpha:
		print('Anova: Models are significantly different (fail to reject H0)',p_anova)
	else:
		print('Anova: Models are not significantly different (reject H0)',p_anova)

else:

	print('None have normal distribution, using non-parametric method Kruskal-Wallis')
	from scipy.stats import kruskal
	stat, p_kruskal = kruskal(model1, model2, model3)
	print('Kruskal: Statistics=%.3f, p=%.3f' % (stat, p_kruskal))
	# interpret
	alpha = 0.05
	if p_kruskal > alpha:
		print('Kruskal: Models are not significantly different (fail to reject H0)',p_kruskal)
	else:
		print('Kruskal: Models are significantly different (reject H0)',p_kruskal)

print('OASIS')

# read csv

csv_path = 'output_oasis.csv'
data_df = pd.read_csv(csv_path, header=None)

# store data in list
model1 = data_df.loc[0,:].tolist()
model2 = data_df.loc[1,:].tolist()
model3 = data_df.loc[2,:].tolist()

# convert data to float

model1 = [float(x) for x in model1]
model2 = [float(x) for x in model2]
model3 = [float(x) for x in model3]

# boxplot

# import matplotlib.pyplot as plt

#plt.figure()
#plt.boxplot([model1,model2,model3])
#plt.show()

# normality test model1

stat, p1 = shapiro(model1)
print('Statistics=%.3f, p=%.3f' % (stat, p1))
# interpret
if p1 > alpha:
	print('Shapiro model1: Model1 looks Gaussian (fail to reject H0)',p1)
else:
	print('Shapiro model1: Model1 does not look Gaussian (reject H0)',p1)

stat, p2 = shapiro(model2)
print('Statistics=%.3f, p=%.3f' % (stat, p2))
# interpret
if p2 > alpha:
	print('Shapiro model2: Model2 looks Gaussian (fail to reject H0)',p2)
else:
	print('Shapiro model2: Model2 does not look Gaussian (reject H0)',p2)

stat, p3 = shapiro(model3)
print('Shapiro: Statistics=%.3f, p=%.3f' % (stat, p3))
# interpret
if p3 > alpha:
	print('Shapiro model3: Model3 looks Gaussian (fail to reject H0)',p3)
else:
	print('Shapiro model3: Model3 does not look Gaussian (reject H0)',p3)

# significancy test
if p1 > alpha and p2 > alpha and p3 > alpha:

	print('All have normal distribution, using parametric method ANOVA')
	from scipy.stats import f_oneway
	stat, p_anova = f_oneway(model1, model2, model3)
	print('Anova: Statistics=%.3f, p=%.3f' % (stat, p_anova))
	# interpret
	if p_anova > alpha:
		print('Anova: Models are significantly different (fail to reject H0)',p_anova)
	else:
		print('Anova: Models are not significantly different (reject H0)',p_anova)

else:

	print('None have normal distribution, using non-parametric method Kruskal-Wallis')
	from scipy.stats import kruskal
	stat, p_kruskal = kruskal(model1, model2, model3)
	print('Kruskal: Statistics=%.3f, p=%.3f' % (stat, p_kruskal))
	# interpret
	alpha = 0.05
	if p_kruskal > alpha:
		print('Kruskal: Models are not significantly different (fail to reject H0)',p_kruskal)
	else:
		print('Kruskal: Models are significantly different (reject H0)',p_kruskal)