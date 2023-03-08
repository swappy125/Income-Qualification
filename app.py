import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image  # To display images
import pickle  # to load model
# from functions import find_target #Function for finding target
# from functions import plot_null_values #Function to check null values
# from functions import replace_yes_no #Function to replace irregularities in data
# from functions import rep_null_val #Function to replace null values
# from functions import drop_columns #Function to drop columns
# from functions import cleaning_pipeline #Function that includes cleaning replacing null values and dropping unnecessary columns
# from functions import data_dictionary # variable dictionary
# from functions import problem_stat # problem statement string
# from functions import steps # Steps to be performed
# from functions import target_inference# target variable inference
# from functions import check_all_member_same_target # as the name suggests it check how many families does not have same target for all variable
# from functions import check_with_head_or_not # checks for how many families are without heads
# from functions import check_no_head_same_target # check for those familie if there is a case where if a family is wothout a head then if the target variable is different for different family members
# from functions import null_val_repl_basis # inference text for null values replacement
# from functions import null_val_replace_logic # null val replecement log text
################################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def Read_data(train):
    # this function reads the file from the current directory
    return pd.read_csv("train.csv")


def find_target(df1, df2):
    """Will be used to find the target variable from the data sets
    #function takes two datasets as input and provides with the column name that is not present in the test set"""
    return (list(set(df1.columns)-set(df2.columns)))[0]


def analyse_catagorical_col(df, col_name, figsize):
    """Function can be used to analyse the categorical features would provide a countplot as an output for the provided (datafarame and column name and a tuple for figuresize) are the expected input variables
    to adjust figsize"""
    plt.figure(figsize=figsize)
    """forms a countplot for each category"""
    sns.countplot(x=df[f'{col_name}'])
    """Set the labels and title of the plot"""
    plt.title(f'{col_name} cardinality distribution')
    """Show the plot"""
    plt.show()


def heat_map_coor_plot(var1, figsize):
    """This function creates a heatmap on the highly correlated features
    takes a correlated matrix as one feature and figsize as s tuple"""
    plt.figure(figsize=figsize)
    sns.heatmap(var1, annot=True, cmap=sns.color_palette("Set2", 8), fmt='.3f')


def plot_null_values(df):
    null_values = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
    null_values = pd.DataFrame(null_values)
    """Reset the index to form it in proper dataframe"""
    null_values.reset_index(inplace=True)
    """Renaming columns"""
    null_values.columns = ['Feature', 'Percentage of missing values']
    return null_values


def plot_top_features(df):
    st.empty()
    missing_values = df.isna().sum().sort_values(ascending=False)
    top10 = missing_values.head(10)
    fig, ax = plt.subplots()
    ax.bar(top10.index, top10.values)
    ax.set_xticklabels(top10.index, rotation=45, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Missing Values')
    ax.set_title('Top 10 Features with Highest Missing Values')
    st.pyplot(fig)


def plot_top_correlation(df):
    correlation = df.corr()
    top_50 = correlation['Target'].abs().sort_values(ascending=False)[1:51]
    return top_50


def plot_top_variance(df):
    variances = np.var(df, axis=0, ddof=1)
    top_15_indices = np.argsort(variances)[::-1][:15]
    top_15_feature_names = np.array(df.columns)[top_15_indices]
    top_15_variances = variances[top_15_indices]
    print("Top 15 features by variance:")
    return top_15_variances


def plot_ownership_status(df):
    st.empty()
    fig, ax = plt.subplots()
    own_variables = [x for x in income_train_df if x.startswith('tipo')]
    df_missing_rent = income_train_df.loc[income_train_df['v2a1'].isnull(
    ), own_variables].sum()
    sns.set_style("darkgrid")
    sns_plot = sns.barplot(x=df_missing_rent.index, y=df_missing_rent.values,
                           color='green', edgecolor='k', linewidth=2, ax=ax)
    sns_plot.set_xticks([0, 1, 2, 3, 4], ['Owns and Paid Off',
                        'Owns and Paying', 'Rented', 'Precarious', 'Other'], rotation=20)
    sns_plot.set(
        title='Home Ownership Status for Households Missing Rent Payments')
    st.pyplot(fig)


def plot_target_count(df):
    st.empty()
    fig, ax = plt.subplots()
    sns.set_style("darkgrid")
    sns_plot = sns.countplot(x='Target', data=income_train_df, ax=ax)
    sns_plot.set(title='Target vs Total_Count')
    st.pyplot(fig)


def plot_unique_values(df):
    unique_values = df.nunique().sort_values(ascending=False)
    unique_values = pd.DataFrame(unique_values)
    """Reset the index to form it in proper dataframe"""
    unique_values.reset_index(inplace=True)
    """Renaming columns"""
    unique_values.columns = ['Feature', 'Count of unique values']
    return unique_values


def plot_outliers_values(df):
    outlier_criteria = 3
    outlier_counts = []
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - outlier_criteria * std
        upper_bound = mean + outlier_criteria * std
        outlier_count = len(
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound)])
        outlier_counts.append(outlier_count)
    outlier_counts.sort(reverse=True)
    outlier_df = pd.DataFrame({'Feature': df.select_dtypes(
        include=np.number).columns, 'Count of outliers': outlier_counts})
    """Reset the index to form it in proper dataframe"""
    outlier_df.reset_index(inplace=True)
    """Renaming columns"""
    outlier_df.drop('index', axis=1, inplace=True)
    return outlier_df


def plot_object_outliers_values(df):
    outlier_counts = []
    for col in df.columns:
        freq_table = df[col].value_counts()
        outliers = freq_table[freq_table < 2]
        outlier_counts.append(len(outliers))
    outlier_df = pd.DataFrame(
        {'Feature': df.columns, 'Count of outliers': outlier_counts})
    """Reset the index to form it in proper dataframe"""
    outlier_df.reset_index(inplace=True)
    """Renaming columns"""
    outlier_df.drop('index', axis=1, inplace=True)
    return outlier_df


def replace_yes_no(df):
    """This function replaces the Yes:1 and No:0 for the following columns and returns
       the dataframe expects the the dataframe as input"""
    mapping = {'yes': 1, 'no': 0}
    for i in df:
        df['dependency'] = df['dependency'].replace(mapping).astype(float)
        df['edjefe'] = df['edjefe'].replace(mapping).astype(float)
        df['edjefa'] = df['edjefa'].replace(mapping).astype(float)
        df['meaneduc'] = df['meaneduc'].replace(mapping).astype(float)
    return df


def check_all_member_same_target(df):
    """This function checks in the data with the help if idhogar column as that is the unique
    identification for each family that the poverty level is same ir not"""
    all_equal = df.groupby('idhogar')['Target'].apply(
        lambda x: x.nunique() == 1)
    not_equal = all_equal[all_equal != True]
    return len(not_equal)


def check_with_head_or_not(df):
    """This function check for the number of families that are without heads"""
    households_head = df.groupby('idhogar')['parentesco1'].sum()
    # Find households without a head
    households_no_head = df.loc[df['idhogar'].isin(
        households_head[households_head == 0].index), :]
    return households_no_head['idhogar'].nunique()


def check_no_head_same_target(df):
    households_head = df.groupby('idhogar')['parentesco1'].sum()
    households_no_head = df.loc[df['idhogar'].isin(
        households_head[households_head == 0].index), :]
    households_no_head_equal = households_no_head.groupby(
        'idhogar')['Target'].apply(lambda x: x.nunique() == 1)
    return sum(households_no_head_equal == False)


def fix_set_poverty_member(df):
    """Below function fixes the target values for all the family member where there are different poverty level for family memebers in the same family"""
    # Groupby the household and figure out the number of unique values
    all_equal = df.groupby('idhogar')['Target'].apply(
        lambda x: x.nunique() == 1)
    # Households where targets are not all equal
    not_equal = all_equal[all_equal != True]
    for household in not_equal.index:
        # Find the correct label (for the head of household)
        true_target = int(df[(df['idhogar'] == household)
                          & (df['parentesco1'] == 1.0)]['Target'])
        # Set the correct label for all members in the household
        df.loc[df['idhogar'] == household, 'Target'] = true_target
    return df


def rep_null_val(df):
    # Replaces all null values the dataframe for following column replacing with 0 as per the finsdings during discovery
    df['v2a1'].fillna(0, inplace=True)
    # """For following column replacing with 0 as per the findings during discovery"""
    df['v18q1'].fillna(0, inplace=True)
    # """For following column replacing with 0 as per the findings during discovery"""
    df['rez_esc'].fillna(0, inplace=True)
    # """For following column replacing with 'edjefe' columns respective value as per the findings during discovery"""
    df['meaneduc'].fillna(df['edjefe'], inplace=True)
    # """For following column replacing with square of 'meaneduc' columns respective value as per the findings during discovery"""
    df['SQBmeaned'].fillna(df['meaneduc']**2, inplace=True)
    return df


def drop_columns(df):
    """This function takes the complete dataframe as input drops the unwanted columns and provides the exactly required dataframe"""
    df.drop(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned',
            'agesq', 'Id', 'idhogar', 'coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total', 'r4t3', 'area2', 'male'], axis=1, inplace=True)
    return df


def cleaning_pipeline(df):
    """Function takes the dataframe then
        1) replaces yes with 1 and no with 0
        2) replaces all the null values in the reuired pattern as per discovery
        3) deletes all the unwanted columns"""
    df1 = replace_yes_no(df)
    df2 = rep_null_val(df1)
    df3 = drop_columns(df2)
    return df3


def problem_stat():
    """DESCRIPTION

Identify the level of income qualification needed for the families in Latin America

# Problem Statement Scenario:
Many social programs have a hard time making sure the right people are given enough aid. It’s tricky when a program focuses on the poorest segment of the population. This segment of population can’t provide the necessary income and expen|se records to prove that they qualify.

In Latin America, a popular method called Proxy Means Test (PMT) uses an algorithm to verify income qualification. With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling or the assets found in their homes to classify them and predict their level of need. While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.

The Inter-American Development Bank (IDB) believes that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance."""


def steps():
    """
# Following actions should be performed:
* Identify the output variable.
* Understand the type of data.
* Check if there are any biases in your dataset.
* Check whether all members of the house have the same poverty level.
* Check if there is a house without a family head.
* Set the poverty level of the members and the head of the house same in a family.
* Count how many null values are existing in columns.
* Remove null value rows of the target variable.
* Predict the accuracy using random forest classifier.
* Check the accuracy using a random forest with cross-validation."""


def data_dictionary():
    """1.	ID = Unique ID
2.	v2a1, Monthly rent payment
3.	hacdor, =1 Overcrowding by bedrooms
4.	rooms, number of all rooms in the house
5.	hacapo, =1 Overcrowding by rooms
6.	v14a, =1 has bathroom in the household
7.	refrig, =1 if the household has a refrigerator
8.	v18q, owns a tablet
9.	v18q1, number of tablets household owns
10.	r4h1, Males younger than 12 years of age
11.	r4h2, Males 12 years of age and older
12.	r4h3, Total males in the household
13.	r4m1, Females younger than 12 years of age
14.	r4m2, Females 12 years of age and older
15.	r4m3, Total females in the household
16.	r4t1, persons younger than 12 years of age
17.	r4t2, persons 12 years of age and older
18.	r4t3, Total persons in the household
19.	tamhog, size of the household
20.	tamviv, number of persons living in the household
21.	escolari, years of schooling
22.	rez_esc, Years behind in school
23.	hhsize, household size
24.	paredblolad, =1 if predominant material on the outside wall is block or brick
25.	paredzocalo, "=1 if predominant material on the outside wall is socket (wood, zinc or absbesto"
26.	paredpreb, =1 if predominant material on the outside wall is prefabricated or cement
27.	pareddes, =1 if predominant material on the outside wall is waste material
28.	paredmad, =1 if predominant material on the outside wall is wood
29.	paredzinc, =1 if predominant material on the outside wall is zink
30.	paredfibras, =1 if predominant material on the outside wall is natural fibers
31.	paredother, =1 if predominant material on the outside wall is other
32.	pisomoscer, "=1 if predominant material on the floor is mosaic, ceramic, terrazo"
33.	pisocemento, =1 if predominant material on the floor is cement
34.	pisoother, =1 if predominant material on the floor is other
35.	pisonatur, =1 if predominant material on the floor is natural material
36.	pisonotiene, =1 if no floor at the household
37.	pisomadera, =1 if predominant material on the floor is wood
38.	techozinc, =1 if predominant material on the roof is metal foil or zink
39.	techoentrepiso, "=1 if predominant material on the roof is fiber cement, mezzanine "
40.	techocane, =1 if predominant material on the roof is natural fibers
41.	techootro, =1 if predominant material on the roof is other
42.	cielorazo, =1 if the house has ceiling
43.	abastaguadentro, =1 if water provision inside the dwelling
44.	abastaguafuera, =1 if water provision outside the dwelling
45.	abastaguano, =1 if no water provision
46.	public, "=1 electricity from CNFL, ICE, ESPH/JASEC"
47.	planpri, =1 electricity from private plant
48.	noelec, =1 no electricity in the dwelling
49.	coopele, =1 electricity from cooperative
50.	sanitario1, =1 no toilet in the dwelling
51.	sanitario2, =1 toilet connected to sewer or cesspool
52.	sanitario3, =1 toilet connected to septic tank
53.	sanitario5, =1 toilet connected to black hole or letrine
54.	sanitario6, =1 toilet connected to other system
55.	energcocinar1, =1 no main source of energy used for cooking (no kitchen)
56.	energcocinar2, =1 main source of energy used for cooking electricity
57.	energcocinar3, =1 main source of energy used for cooking gas
58.	energcocinar4, =1 main source of energy used for cooking wood charcoal
59.	elimbasu1, =1 if rubbish disposal mainly by tanker truck
60.	elimbasu2, =1 if rubbish disposal mainly by botan hollow or buried
61.	elimbasu3, =1 if rubbish disposal mainly by burning
62.	elimbasu4, =1 if rubbish disposal mainly by throwing in an unoccupied space
63.	elimbasu5, "=1 if rubbish disposal mainly by throwing in river, creek or sea"
64.	elimbasu6, =1 if rubbish disposal mainly other
65.	epared1, =1 if walls are bad
66.	epared2, =1 if walls are regular
67.	epared3, =1 if walls are good
68.	etecho1, =1 if roof are bad
69.	etecho2, =1 if roof are regular
70.	etecho3, =1 if roof are good
71.	eviv1, =1 if floor are bad
72.	eviv2, =1 if floor are regular
73.	eviv3, =1 if floor are good
74.	dis, =1 if disable person
75.	male, =1 if male
76.	female, =1 if female
77.	estadocivil1, =1 if less than 10 years old
78.	estadocivil2, =1 if free or coupled uunion
79.	estadocivil3, =1 if married
80.	estadocivil4, =1 if divorced
81.	estadocivil5, =1 if separated
82.	estadocivil6, =1 if widow/er
83.	estadocivil7, =1 if single
84.	parentesco1, =1 if household head
85.	parentesco2, =1 if spouse/partner
86.	parentesco3, =1 if son/doughter
87.	parentesco4, =1 if stepson/doughter
88.	parentesco5, =1 if son/doughter in law
89.	parentesco6, =1 if grandson/doughter
90.	parentesco7, =1 if mother/father
91.	parentesco8, =1 if father/mother in law
92.	parentesco9, =1 if brother/sister
93.	parentesco10, =1 if brother/sister in law
94.	parentesco11, =1 if other family member
95.	parentesco12, =1 if other non family member
96.	idhogar, Household level identifier
97.	hogar_nin, Number of children 0 to 19 in household
98.	hogar_adul, Number of adults in household
99.	hogar_mayor, # of individuals 65+ in the household
100.	hogar_total, # of total individuals in the household
101.	dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
102.	edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
103.	edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
104.	meaneduc,average years of education for adults (18+)
105.	instlevel1, =1 no level of education
106.	instlevel2, =1 incomplete primary
107.	instlevel3, =1 complete primary
108.	instlevel4, =1 incomplete academic secondary level
109.	instlevel5, =1 complete academic secondary level
110.	instlevel6, =1 incomplete technical secondary level
111.	instlevel7, =1 complete technical secondary level
112.	instlevel8, =1 undergraduate and higher education
113.	instlevel9, =1 postgraduate higher education
114.	bedrooms, number of bedrooms
115.	overcrowding, # persons per room
116.	tipovivi1, =1 own and fully paid house
117.	tipovivi2, "=1 own, paying in installments"
118.	tipovivi3, =1 rented
119.	tipovivi4, =1 precarious
120.	tipovivi5, "=1 other(assigned, borrowed)"
121.	computer, =1 if the household has notebook or desktop computer
122.	television, =1 if the household has TV
123.	mobilephone, =1 if mobile phone
124.	qmobilephone, # of mobile phones
125.	lugar1, =1 region Central
126.	lugar2, =1 region Chorotega
127.	lugar3, =1 region PacÃfico central
128.	lugar4, =1 region Brunca
129.	lugar5, =1 region Huetar AtlÃ¡ntica
130.	lugar6, =1 region Huetar Norte
131.	area1, =1 zoinformationna urbana
132.	area2, =2 zona rural
133.	age= Age in years
134.	SQBescolari= escolari squared
135.	SQBage, age squared
136.	SQBhogar_total, hogar_total squared
137.	SQBedjefe, edjefe squared
138.	SQBhogar_nin, hogar_nin squared
139.	SQBovercrowding, overcrowding squared
140.	SQBdependency, dependency squared
141.	SQBmeaned, square of the mean years of education of adults (>=18) in the household
142.	agesq= Age squared"""


def target_inference():
    """From above we can observe that the training data is biased as the model will
get the opportunity learn very few Extreme poverty cases. And that can lead to a state where the
model not identify the Extreme poor cases at all
    """


def null_val_repl_basis():
    """##### Inference
Looking at the different types of data and null values for each feature
We found the following:
*   No null values for Integer type features.
*   No null values for object type features.
##### For float64 types below features has null value
*   v2a1 6860 ==>71.77%  variable explaination => Monthly rent payment
*   v18q1 7342 ==>76.82% variable explaination => Number of tablets household owns
*   rez_esc 7928 ==>82.95% variable explaination => Years behind in school
*   meaneduc 5 ==>0.05% variable explaination => average years of education for adults (18+)
*   SQBmeaned 5 ==>0.05% variable explaination => square of the mean years of education of adults (>=18) in the household
"""


def null_val_replace_logic():
    """ Null Value Replacement logic
    v2a1
    #From above this is observed that the when house is fully paid there are no furthers monthly rents
    #Lets add 0 for all monhtly rents
    v18q1
    #Looking at the above data it makes sense that when owns a tablet column is 0, there will be no number of tablets household owns. Lets add 0 for all the null values.
    rez_esc
    #We can observe that when min age is 7 and max age is 17 for Years, then the 'behind in school' there is only on values that is missing.Lets add 0 for the null values.
    meaneduc
    #From above outputs we infer that - There are five datapoints with meaneduc as NaN. And all have 18+ age. The value of meaneduc feature is same as 'edjefe' if the person is male and 'edjefa' if the person is female for majority of datapoints.
    Hence, we treat the 5 NaN values by replacing the with respective 'edjefe'.
    SQBmeaned
    #Square of the mean years of education of adults (>=18) in the household - 5 values hence lets replace the null values by respective ['meaneduc']**2
    **************************************************************************************************************************************************************************************************************************************************
    To fix the irregular columns we need to map yes:1||no:0
"""


################################################################################################################################
# load model
model = pickle.load(open('model.pkl', 'rb'))
################################################################################################################################
# functions
# To Improve speed and cache data


@st.cache_data(persist=True)
def Read_data(filename):
    """Function reads csv data and returns as a dataframe"""
    return pd.read_csv(filename)


################################################################################################################################
# reading data
income_train_df = Read_data('train.csv')
income_test_df = Read_data('test.csv')
###############################################################################################################################
# Separating columns in different data types
float_columns = income_train_df.select_dtypes('float').columns.tolist()
int_columns = income_train_df.select_dtypes('int').columns.tolist()
object_columns = income_train_df.select_dtypes('object').columns.tolist()
# separating columns which indicate home ownership
own_variables = [x for x in income_train_df if x.startswith('tipo')]
###############################################################################################################################
# Title and Subheader


def main():  # beginning of the statement execution
    html_temp = """
    <<div style="background-image: url(https://www.sbs.com.au/topics/sites/sbs.com.au.topics/files/poor_rich.jpg);
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    padding:150px">
    <h2 style="color:red;text-align:center;">Income Qualifications</h2>
    </div>




    """

    st.markdown(html_temp, unsafe_allow_html=True)

    st.text("Team: Naaz,Tanay,Swapnil,Sagar,Vijay")
    st.text("Mentor: Shriti Datta")
    if st.sidebar.checkbox('Problem Statement'):
        problem_stat
    if st.sidebar.checkbox('Show Data Dictionary'):
        data_dictionary
    if st.sidebar.checkbox('Steps To be performed'):
        steps
    if st.sidebar.checkbox('Finding Target Variable and Checking for Bias'):
        Target = find_target(income_train_df, income_test_df)
        st.text(
            f"The {find_target(income_train_df,income_test_df)} is the Output Variable")
        st.subheader(f"{Target} variable cardinal distribution")
        st.write(
            (income_train_df[Target].value_counts()/len(income_train_df))*100)
        st.subheader("Mappings")
        """
             * 1 : Extreme Poverty
             * 2 : Moderate Poverty
             * 3 : Vulnerable Households
             * 4 : Non-vulnerable Households"""
        st.subheader("Inference")
        target_inference
    if st.sidebar.checkbox('Check Dataset information'):
        st.subheader("Dataset Sample")
        st.write(income_train_df.iloc[:50])
        st.subheader("Dataset Shape")
        st.text(
            f"The Training Dataset Consists of {income_train_df.shape[0]} rows and {income_train_df.shape[1]} columns")
        st.subheader("Dataset Description")
        st.write(income_train_df.describe())
        st.subheader("Columns Info")
        data_dim = st.radio("Column type information",
                            ("float_columns", "int_columns", "object_columns"))
        if data_dim == "float_columns":
            st.text(
                f'Out of 143 {len(float_columns)} are Float_type type columns')
            st.write(float_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[float_columns]))
            st.subheader("Unique values in each column")
            st.write(plot_unique_values(income_train_df[float_columns]))
            st.subheader("No of outliers")
            st.write(plot_outliers_values(income_train_df[float_columns]))
            st.subheader("Inference")
            st.text(
                """                 *   v2a1 6860 ==>71.77%  variable explaination => Monthly rent payment
                *   v18q1 7342 ==>76.82% variable explaination => Number of tablets household owns
                *   rez_esc 7928 ==>82.95% variable explaination => Years behind in school
                *   meaneduc 5 ==>0.05% variable explaination => average years of education for adults (18+)
                *   SQBmeaned 5 ==>0.05% variable explaination => square of the mean years of education of adults""")
        if data_dim == "object_columns":
            st.text(
                f'Out of 143 {len(object_columns)} are Categorical type columns')
            st.write(object_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[object_columns]))
            st.subheader("Unique values in each column")
            st.write(plot_unique_values(income_train_df[object_columns]))
            st.subheader("No of outliers")
            st.write(plot_object_outliers_values(
                income_train_df[object_columns]))
            st.subheader("Inference")
            st.text('There Are no null values in object columns')
        if data_dim == "int_columns":
            st.text(f'Out of 143 {len(int_columns)} are Integer type columns')
            st.write(int_columns)
            st.subheader("Null Values Status")
            st.write(plot_null_values(income_train_df[int_columns]))
            st.subheader("Unique values in each column")
            st.write(plot_unique_values(income_train_df[int_columns]))
            st.subheader("No of outliers")
            st.write(plot_outliers_values(income_train_df[int_columns]))
            st.subheader("Inference")
            st.text('There Are no null values in integer type columns')

        st.subheader("Top 10 Features with highest missing values")
        st.write(plot_top_features(income_train_df))

        st.subheader("Duplicate Rows")
        duplicates = income_train_df.duplicated().sum()
        st.subheader(
            f'There are {duplicates} duplicate rows in dataset')

        st.subheader("Target Count")
        target_count = income_train_df.Target.value_counts()
        st.write(target_count)
        st.write(plot_target_count(income_train_df))

        st.subheader("House Ownership Status")
        st.write(plot_ownership_status(income_train_df))

        st.subheader("Top 50 Highest Correlated Features")
        st.write(plot_top_correlation(income_train_df))

        st.subheader("Top 15 Highest Variance Features")
        st.write(plot_top_variance(income_train_df))

    if st.sidebar.checkbox("Check Family members information"):
        st.subheader(
            "Check if all family members have the same poverty level or not")
        st.text(
            f'There are {check_all_member_same_target(income_train_df)} households where the family members do not have same poverty level')
        st.subheader("Check if all families are with Heads")
        st.text(
            f'There are {check_with_head_or_not(income_train_df)} families without family Head')
        st.subheader(
            "Check if families without Heads have different poverty levels")
        st.text(
            f"There are {check_no_head_same_target(income_train_df)} families with no Heads and different poverty levels")
        st.subheader("Inference")
        st.text(
            "There are 85 Families which have different poverty levels but all of them have Family heads")
    if st.sidebar.checkbox("Basis for Null Value replacement"):
        null_val_repl_basis
        st.text(
            f"column {float_columns[0]} has {income_train_df[float_columns[0]].isnull().sum()} null values")
        # Plot of the home ownership variables for home missing rent payments
        plt.rcParams["figure.figsize"] = (3, 3)
        fig, x = plt.subplots()
        income_train_df.loc[income_train_df['v2a1'].isnull(
        ), own_variables].sum().sort_values().plot(kind='bar')
        plt.xticks([0, 1, 2, 3, 4], ['Owns and Paying', 'Rented',
                   'Precarious',  'Other', 'Owns and Paid Off'], rotation=90)
        plt.title('Home Ownership Status for Households Missing Rent', size=5)
        # plotting figure
        st.pyplot(fig)
        st.text(
            f"colum {float_columns[1]} has {income_train_df[float_columns[1]].isnull().sum()} null values")
        heads = income_train_df.loc[income_train_df['parentesco1'] == 1].copy()
        heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
        """Lets look at v18q1 (total nulls: 7342) : number of tablets household owns
        #why the null values, Lets look at few rows with nulls in v18q1
        #Columns related to number of tablets household owns
        #v18q, owns a tablet
        #Since this is a household variable, it only makes sense to look at it on a household level
        #so we'll only select the rows for the head of household."""
        fig, x = plt.subplots()
        income_train_df[float_columns[1]].value_counts(
        ).sort_index().plot(kind='bar')
        plt.xlabel(float_columns[0])
        plt.ylabel('Value_counts')
        # plotting figure
        st.pyplot(fig)
        st.text(
            f"colum {float_columns[2]} has {income_train_df[float_columns[2]].isnull().sum()} null values")
        """Years behind in school
        #why the null values, Lets look at few rows with nulls in rez_esc
        #Columns related to Years behind in school
        #Age in years
        # Lets look at the data with not null values first."""
        st.text("Now grouping rez_esc age to check the pattern")
        st.write(
            income_train_df[income_train_df['rez_esc'].notnull()]['age'].describe())
        """There is one value that has Null for the 'behind in school' column with age between 7 and 17"""
        st.write(income_train_df.loc[(income_train_df['rez_esc'].isnull() &
                                      ((income_train_df['age'] > 7) & (income_train_df['age'] < 17)))]['age'].describe())
        """There is only one member in household for the member with age 10 and who is 'behind in school'. This explains why the member is
        behind in school."""
        st.write(income_train_df[(income_train_df['age'] == 10)
                 & income_train_df['rez_esc'].isnull()].head())
        st.text(
            f"colum {float_columns[3]} has {income_train_df[float_columns[3]].isnull().sum()} null values")
        st.write(income_train_df[income_train_df['meaneduc'].isnull()].loc[:, ['age', 'meaneduc', 'edjefe', 'edjefa', 'instlevel1',
                 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']])
        st.text(
            f"colum {float_columns[7]} has {income_train_df[float_columns[7]].isnull().sum()} null values")
        st.write(income_train_df[income_train_df['SQBmeaned'].isnull()].loc[:, [
                 'SQBmeaned', 'meaneduc', 'edjefe', 'edjefa', 'instlevel1', 'instlevel2']])
        st.text("We have also found that there are few columns where 'yes' should be mapped with 1 and 0 with 'no' as other values follow the same pattern")
        """'dependency' column"""
        st.write(income_train_df['dependency'].value_counts())
        st.subheader("Inference")
        """'edjefe' column"""
        st.write(income_train_df['edjefe'].value_counts())
        """'edjefa' column"""
        st.write(income_train_df['edjefa'].value_counts())
        """'meaneduc' column"""
        st.write(income_train_df['meaneduc'].value_counts())

        st.subheader("Inference")
        null_val_replace_logic
    if st.sidebar.checkbox("Feature Selection"):
        st.subheader("Columns Break Up")
        id_ = ['Id', 'idhogar', 'Target']
        ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3',
                    'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7',
                    'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5',
                    'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10',
                    'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3',
                    'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
                    'instlevel9', 'mobilephone']
        ind_ordered = ['rez_esc', 'escolari', 'age']
        hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo',
                   'paredpreb', 'pisocemento', 'pareddes', 'paredmad',
                   'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother',
                   'pisonatur', 'pisonotiene', 'pisomadera',
                   'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo',
                   'abastaguadentro', 'abastaguafuera', 'abastaguano',
                   'public', 'planpri', 'noelec', 'coopele', 'sanitario1',
                   'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
                   'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
                   'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4',
                   'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
                   'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3',
                   'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',
                   'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
                   'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']
        hh_ordered = ['rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1',  'r4t2',
                      'r4t3', 'v18q1', 'tamhog', 'tamviv', 'hhsize', 'hogar_nin',
                      'hogar_adul', 'hogar_mayor', 'hogar_total',  'bedrooms', 'qmobilephone']
        hh_cont = ['v2a1', 'dependency', 'edjefe',
                   'edjefa', 'meaneduc', 'overcrowding']
        st.text(
            f'id columns==> {id_} \n with 90% unique value or is target variable')
        st.text(
            f'ind_bool columns==> {ind_bool} \n with 0|1 value for an individual')
        st.text(
            f'ind_ordered columns==> {ind_ordered} \n with ordered numerical value')
        st.text(
            f'hh_bool columns==> {hh_bool} \n with 0|1 value for house hold')
        st.text(
            f'hh_cont columns==> {hh_cont} \n with house hold continous values')
        st.subheader("Cheking for multicollinearity")
        st.text("Checking for redundant household variables select the ones with 'parentesco1' value as 1 only remove the ones which does not have a head of family")
        "Taking families with heads only"
        heads = income_train_df.loc[income_train_df['parentesco1'] == 1, :]
        heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
        corr_matrix = heads.corr()
        "Correlation Matrix"
        st.write(corr_matrix)
        "Selecting the upper traingle of corr_matrix"
        # Selecting the upper traingle of corr_matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        plt.rcParams["figure.figsize"] = (8, 8)
        fig, x = plt.subplots()
        sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs(
        ) > 0.9], annot=True, cmap=sns.color_palette("Set2", 8), fmt='.3f')
        st.write(fig)
        cols_to_drop = [column for column in upper.columns if any(
            abs(upper[column]) > 0.95)]
        st.text(
            f'Columns to drop {cols_to_drop} \n as they have above 0.95 correlation')
        # st.write()
        """
#### Inference
There are several variables here having to do with the size of the house:
*   r4t3, Total persons in the household
*   tamhog, size of the household
*   tamviv, number of persons living in the household
*   hhsize, household size
*   hogar_total, # of total individuals in the household
*   These variables are all highly correlated with one another.
*   Removing the male as well, as this would not be needed in model creation
*   Removing the Id and 'idhogar' as well, as this would not be needed in model creation
##### There are some Squared Variables and we understand that these would not add any value to the classification model.
##### Hence dropping these features -
*   'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq','Id','idhogar','coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total', 'r4t3', 'area2', 'male'"""
    # if st.sidebar.checkbox("Model Building"):
    #     st.text("Random Forrest Classifier was selected as per instructions")
    #     image = Image.open('Model 1 performance.jpg')
    #     image2 = Image.open('feature imp.jpg')
    #     st.image(image, caption='Model1 performance')
    #     st.image(image2, caption='Feature Importance')
    #     """From above figure, meaneduc, dependency, overcrowding has significant influence on the model."""
    #     st.text("Post Cross-Validtion and taking 150 trees the \n accuracy scores average increased from \n 94.25547753700772 to 94.30783352929198")
    if st.sidebar.checkbox("Predictions"):
        """Upload csv file for feeding data to model"""
        data_upload = st.file_uploader("Upload File", type=[".csv"])
        dftest = pd.read_csv(data_upload)
        if data_upload is not None:
            st.subheader("Test Dataset Sample")
            st.write(dftest)
            dftest = cleaning_pipeline(dftest)
            predictions = model.predict(dftest)
            st.subheader("Prediction for Test Dataset")
            df = pd.DataFrame(dftest, columns=list(dftest.columns))
            df['predictions'] = predictions
            st.write(df)
            st.subheader("predictions variable cardinal distribution")
            st.write(
                (df['predictions'].value_counts()/len(df))*100)


if __name__ == '__main__':  # check for main executed when programme is called
    main()
