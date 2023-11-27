# In this I have performed the steps for Data pre processing On DataCoSupplyChain
import pandas as pd

# to import csv file
task_data = pd.read_csv(r"D:/Data Science training/EDA/material/tasks/DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')

task_data.info() # to see number of rows and columns and types of data 

# dropping of unwanted columns and checking for missing values
task_data.isna().sum()
task_data.drop(["Days for shipping (real)", "Late_delivery_risk", "Category Id", "Customer Email", "Customer Id", "Customer Password", 
                "Customer State", "Customer Street", "Customer Zipcode", "Department Id", "Latitude", "Longitude", "Market",
                "Order Customer Id", "Order Item Cardprod Id", "Order Item Discount Rate", "Order Item Id", "Order Item Profit Ratio",
                "Order Region", "Order Zipcode", "Product Card Id", "Product Category Id", 
                "Product Description", "Product Image"],
               axis=1, inplace = True) # dropping of column

task_data.info()

# type casting

task_data["order date (DateOrders)"] = task_data["order date (DateOrders)"].astype('datetime64')

task_data["shipping date (DateOrders)"] = task_data["shipping date (DateOrders)"].astype('datetime64')

task_data.dtypes


# droping of duplicates

# here we are checking duplicates in rows
duplicates = task_data.duplicated()
duplicates # here it will tell whether the row is duplicate of other row or not in form of true or false

sum(duplicates) 

# checking cor-relation between them
correlation = task_data.corr() 
# in correlation we can see there strong correlation between sales per customer and order item total, benefit per order and order profit per order
task_data.drop(["Order Item Total", "Order Profit Per Order", "Product Price"], axis = 1, inplace = True)


# Outlier treatment

task_data.dtypes # to plot boxplots, checking data types

import seaborn as sns

sns.boxplot(task_data["Days for shipment (scheduled)"]) # no outliers

sns.boxplot(task_data["Benefit per order"])

sns.boxplot(task_data["Sales per customer"])

sns.boxplot(task_data["Order Item Discount"])

sns.boxplot(task_data["Order Item Product Price"])

sns.boxplot(task_data["Order Item Quantity"]) # no outliers

sns.boxplot(task_data["Sales"])

sns.boxplot(task_data["Product Status"]) # no outliers

# Winsorization

from feature_engine.outliers import Winsorizer

winsor_iqr = Winsorizer(capping_method = "iqr",
                        tail = "both",
                        fold = 1.5,
                        variables = ["Benefit per order", "Sales per customer", "Order Item Discount", "Order Item Product Price",
                                     "Sales"])

task_data = winsor_iqr.fit_transform(task_data)

sns.boxplot(task_data["Benefit per order"])

sns.boxplot(task_data["Sales per customer"])

sns.boxplot(task_data["Order Item Discount"])

sns.boxplot(task_data["Order Item Product Price"])

sns.boxplot(task_data["Sales"])


# zero variance and near zero variance

task_data.var(axis=0) == 0

task_data.drop(["Product Status"], axis=1, inplace = True)



# Discretization


task_data["Days for shipment (scheduled)"].describe()

task_data["Days for shipment (scheduled)"] = pd.cut(task_data["Days for shipment (scheduled)"],
                                                    bins = [min(task_data["Days for shipment (scheduled)"]),
                                                            task_data["Days for shipment (scheduled)"].mean(),
                                                            max(task_data["Days for shipment (scheduled)"])],
                                                    include_lowest = True,
                                                    labels = ["fast", "slow"])
task_data["Days for shipment (scheduled)"].value_counts()

task_data["Benefit per order"].describe()

task_data["Benefit per order"] = pd.cut(task_data["Benefit per order"],
                                        bins = [min(task_data["Benefit per order"]),
                                                task_data["Benefit per order"].mean(),
                                                max(task_data["Benefit per order"])],
                                        include_lowest = True,
                                        labels = ["low benefit", "high benefit"])
task_data["Benefit per order"].value_counts()


task_data["Sales per customer"].describe()

task_data["Sales per customer"] = pd.cut(task_data["Sales per customer"],
                                         bins = [min(task_data["Sales per customer"]),
                                                 task_data["Sales per customer"].quantile(0.50),
                                                 max(task_data["Sales per customer"])],
                                         include_lowest = True,
                                         labels = ["low sale", "high sale"])
task_data["Sales per customer"].value_counts()

task_data["Order Item Quantity"].describe()

task_data["Order Item Quantity"] = pd.cut(task_data["Order Item Quantity"],
                                          bins = [min(task_data["Order Item Quantity"]),
                                                  task_data["Order Item Quantity"].mean(),
                                                  max(task_data["Order Item Quantity"])],
                                          include_lowest = True,
                                          labels = ["low quantity", "high quantity"])
task_data["Order Item Quantity"].value_counts()





# Dummy variables

task_data.dtypes

# merging two columns into one column
task_data["name"] = task_data["Customer Fname"]+ " "  +task_data["Customer Lname"]

task_data.drop(["Customer Fname", "Customer Lname"], axis = 1, inplace = True)

# Execute the below line to get csv file to perform data analysis and to get insights in PowerBI
# task_data.to_csv("D:/Data Science training/EDA/material/tasks/Dataforpowerbi.csv", encoding='utf-8', index=False)

task_data.columns

# re-arranging columns
task_data = task_data[["name", "order date (DateOrders)", "Order Id", "Order Item Discount", "Order Item Product Price",
                       "Sales", "shipping date (DateOrders)", "Type", "Days for shipment (scheduled)", "Benefit per order",
                       "Sales per customer", "Delivery Status", "Category Name", "Customer City", "Customer Country",
                       "Customer Segment", "Department Name", "Order City", "Order Country", "Order Item Quantity",
                       "Order State", "Order Status", "Product Name", "Shipping Mode"]]

# Extract the columns you want to create dummies for into a new DataFrame
selected_columns = task_data[["Category Name", "Department Name"]]

# Use pd.get_dummies to create dummy variables for these columns
dummies = pd.get_dummies(selected_columns, drop_first=True)

dummies.dtypes
corre = dummies.corr() # Here there are some columns with
dummies.drop(["Department Name_Book Shop", "Department Name_Pet Shop"], axis=1, inplace = True) 


# Add the dummy columns back to the original DataFrame
task_data = pd.concat([task_data, dummies], axis=1)

# Optionally, you can drop the original columns if needed
task_data.drop(["Category Name", "Department Name"], axis=1, inplace=True)




# label encoding
from sklearn.preprocessing import LabelEncoder

#creating instance of label encoder
encoder = LabelEncoder()

# columns to encode
columns_encode = ["Type", "Days for shipment (scheduled)", "Benefit per order",
                  "Sales per customer", "Delivery Status", "Customer City", 
                  "Customer Country", "Customer Segment", "Order City", "Order Country",
                  "Order Item Quantity", "Order State", "Order Status", "Product Name", "Shipping Mode"]

for column in columns_encode:
    task_data[column] = encoder.fit_transform(task_data[column])

                                                                                         
                                                                               



# Normal Quantile-Quantile plot
# Transformation
import seaborn as sns
import scipy.stats as stats
import pylab
import matplotlib.pyplot as plt


# For Order Item Discount
# original data
plt.figsize = (10,6)
stats.probplot(task_data["Order Item Discount"], dist = "norm", plot = pylab)
plt.title("Order Item Discount")
plt.show()

from feature_engine import transformation

# Set up the variable transformer
tf = transformation.YeoJohnsonTransformer(variables = 'Order Item Discount')

task_data = tf.fit_transform(task_data)

# Transformed data
plt.figsize = (10,6)
stats.probplot(task_data["Order Item Discount"], dist = "norm", plot = pylab)
plt.title("Order Item Discount")
plt.show()



# For Order Item Product Price
# original data
plt.figsize = (10,6)
stats.probplot(task_data["Order Item Product Price"], dist = "norm", plot = pylab)
plt.title("Order Item Product Price")
plt.show()

# Plotting the original data (non-normal)
plt.figure(figsize=(10,6))
sns.distplot(task_data["Order Item Product Price"], color = "green", hist = False, kde = True)
plt.title(" Order Item Product Price ")
plt.show()

# Transform training data & save lambda value
task_data["Order Item Product Price"], fitted_lambda = stats.boxcox(task_data["Order Item Product Price"])

# Plotting the fitted data (normal)
sns.distplot(task_data["Order Item Product Price"], hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Normal", color = "green")
# adding legends to the subplots
plt.legend(loc = "upper right")

print(f"Lambda value used for Transformation: {fitted_lambda}")

# transformed data
plt.figsize = (10,6)
stats.probplot(task_data["Order Item Product Price"], dist = "norm", plot = pylab)
plt.title("Order Item Product Price")
plt.show()



# For Sales
# Original data
plt.figsize = (10,6)
stats.probplot(task_data["Sales"], dist = "norm", plot = pylab)
plt.title("Sales")
plt.show()

# Plotting the original data (non-normal)
plt.figure(figsize=(10,6))
sns.distplot(task_data["Sales"], color = "violet", hist = False, kde = True)
plt.title(" Sales ")
plt.show()

# Transform training data & save lambda value
task_data["Sales"] , lambda_value = stats.boxcox(task_data["Sales"])

# Plotting the original data () 

sns.distplot(task_data["Sales"], hist = False, kde = True,
             kde_kws = {'shade' : True, 'linewidth' : 4},
             label = "Normal", color = "violet")
# adding legends to the subplots
plt.legend(loc = "upper right")

print(f"Lambda value used for Transformation: {lambda_value}")




######## Standardization and Normalization #########


a = task_data.describe()

from sklearn.preprocessing import StandardScaler

standard = StandardScaler()

columns_to_standard = [["Order Item Discount", "Order Item Product Price", "Sales", "Type", "Days for shipment (scheduled)", 
                       "Benefit per order", "Sales per customer", "Delivery Status", "Customer City", 
                       "Customer Country", "Customer Segment", "Order City", "Order Country",
                       "Order Item Quantity", "Order State", "Order Status", "Product Name", "Shipping Mode"]]

for column in columns_to_standard:
    task_data[column] = pd.DataFrame(standard.fit_transform(task_data[column]))

b = task_data.describe()





























