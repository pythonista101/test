import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





# importing
data = pd.read_csv(
    "Data_Science_Fields_Salary_Categorization.csv",
    index_col=0
)
codes = pd.read_excel('codes.xlsx')

raw = data.copy()

# dtypes
data.Salary_In_Rupees = data.Salary_In_Rupees.str.replace(',', '').astype('float')
data.Working_Year = data.Working_Year.astype('int64')

eur_to_rup = {
    2020: 85.22,
    2021: 88.24,
    2022: 82.76
}


def conversion(rup, year):
    return round(rup / eur_to_rup[year], 2)


vconv = np.vectorize(conversion)

data['Salary_eur'] = vconv(data.Salary_In_Rupees, data.Working_Year)

data = data.merge(codes, how='left', left_on='Company_Location', right_on='2_digit')

gdp = pd.read_csv('gdp_per_capita.csv')
d = gdp[['Country Name', 'Code', '2020']].rename(columns = {'2020':'BBP per hoofd'}).dropna()
gdp = gdp[['Code', '2020']]





data = data.merge(gdp, how='left', left_on='3_digit', right_on='Code')

usd_to_eur = 0.877  # average exchange rate 2020

data = data.rename(columns={'2020': 'gdp_euro'})
data["gdp_euro"] = usd_to_eur * data["gdp_euro"]
data["gdp_euro"] = np.around(data["gdp_euro"], 2)
data['percentage'] = np.around(data['Salary_eur'] / data['gdp_euro'] * 100, 2)

data['expat'] = np.where(data.Employee_Location == data.Company_Location, 'No', 'Yes')

data = data[['Working_Year', 'Designation', 'Experience', 'Employment_Status',
             'Employee_Location', 'Company_Location', 'expat',
             'Company_Size', 'Remote_Working_Ratio', 'Salary_eur',
             'gdp_euro', 'percentage', 'Country']]
data_kasper = data.copy()

x_kasper=pd.DataFrame(pd.Series(data_kasper.groupby('Designation').count().sort_values('Working_Year', ascending=False).index))
y_kasper=pd.DataFrame(data_kasper.groupby('Designation')['Salary_eur'].mean())
final_kasper=x_kasper.merge(y_kasper, on='Designation', how='left')

data = data.rename(columns={
    'Working_Year': 'Jaar',
    # 'Designation':'Beroep',
    # 'Experience':'Ervaring',
    'Employment_Status': 'Status',
    'Employee_Location': 'Land persoon',
    'Company_Location': 'Land bedrijf',
    'Company_Size': 'Grootte',
    'Remote_Working_Ratio': 'Thuiswerk ratio',
    'Salary_eur': 'Salaris €',
    'gdp_euro': 'GDP €',
    'percentage': '%',
    'Country': 'Land'
})


def encoder(dataframe, column):
    col = dataframe[column]

    trans = {}
    count = 0
    for a in col:
        if a not in trans.keys():
            trans[a] = count
            count += 1

    copyseries = col.copy()
    for index, element in enumerate(col):
        copyseries[index] = trans[element]

    dataframe[column] = copyseries

    backwards = {}

    for a in trans:
        value = trans[a]
        backwards[value] = a

    return trans, backwards


train_data = data.copy()

desig_fwd, desig_bwd = encoder(train_data, 'Designation')
status_fwd, status_bwd = encoder(train_data, 'Status')
ex_fwd, ex_bwd = encoder(train_data, 'Experience')
cp_fwd, cp_bwd = encoder(train_data, 'Land persoon')
cc_fwd, cc_bwd = encoder(train_data, 'Land bedrijf')
expat_fwd, expat_bwd = encoder(train_data, 'expat')
size_fwd, size_bwd = encoder(train_data, 'Grootte')

train_data = train_data[['Jaar', 'Designation', 'Experience', 'Status', 'Land persoon',
                         'Land bedrijf', 'expat', 'Grootte', 'Thuiswerk ratio',
                         'GDP €', '%', 'Salaris €']]

X = train_data[['Designation', 'Experience', 'Status', 'Land persoon',
                'Land bedrijf', 'expat', 'Grootte', 'Thuiswerk ratio']]
Y = train_data[['%', 'Salaris €']]

from sklearn import tree

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, Y)


def salary_prediction(des, ex, stat, la_pe, la_be, size, rat):
    if la_pe == la_be:
        expat = 'No'
    else:
        expat = 'Yes'

    x = pd.DataFrame({
        'Designation': [desig_fwd[des]],
        'Experience': [ex_fwd[ex]],
        'Status': [status_fwd[stat]],
        'Land persoon': [cp_fwd[la_pe]],
        'Land bedrijf': [cc_fwd[la_be]],
        'expat': [expat_fwd[expat]],
        'Grootte': [size_fwd[size]],
        'Thuiswerk ratio': [rat]
    })
    return clf.predict(x)[0]


# some variabelen

stat_dict = {'Fulltime': 'FT', 'Parttime': 'PT', 'Oproepbasis': 'CT', 'Freelance': 'FL'}
ex_dict = {'Entry': 'EN', 'Medior': 'MI', 'Senior/Expert': 'SE', 'Executive': 'EX'}
size_dict = {"Groot": 'L', "Medium": 'M', "Klein": 'S'}

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(data['Salaris €'])

from PIL import Image

salary_image = Image.open('salaris_ss.png')
intro_image = Image.open('Intro afbeelding.png')
qrcode = Image.open('QRcode.png')
appeltaart = Image.open('piechart.png')




with open('titel.txt', 'r') as file:
    t_titel = file.read().replace('\n', '')

with open('introductie.txt', 'r') as file:
    t_introductie = file.read()
with open('feature_engineering1.txt', 'r') as file:
    t_feature_1 = file.read()
with open('feature_engineering2.txt', 'r') as file:
    t_feature_2 = file.read()
with open('feature_engineering3.txt', 'r') as file:
    t_feature_3 = file.read()
with open('Discussie.txt', 'r') as file:
    Discussie = file.read()
with open('api.txt', 'r') as file:
    api = file.read()
with open('Discussie2.txt', 'r') as file:
    Discussie2 = file.read()
with open('wereldbol.txt', 'r') as file:
    wereldbol = file.read()
    
with open('Dataanalyse.txt', 'r') as file:
    Dataanalyse = file.read()
with open('Visualisatie.txt', 'r') as file:
    Visualisatie = file.read()
with open('UnivariateData.txt', 'r') as file:
    UnivariateData = file.read()
with open('brandon1.txt', 'r') as file:
    brandon1 = file.read()

import plotly.express as px
gdp_map_bol = px.choropleth(d, locations='Code', color='BBP per hoofd',
                        projection='orthographic')
gdp_map_plat = px.choropleth(d, locations='Code', color='BBP per hoofd',
                        projection='equirectangular')
gdp_treemap= px.treemap(d, path = ['Country Name', 'BBP per hoofd'], values = 'BBP per hoofd', color = 'BBP per hoofd')



fig2, ax3 = plt.subplots()
ax3.grid()
ax3.bar(eur_to_rup.keys(), eur_to_rup.values())

min = 80
max = 90
ax3.set_ylim(min, max)
ax3.set_xticks([2020,2021,2022])
yticks = np.linspace(max,min,8)
ax3.set_yticks(yticks)
ax3.set_title('Roepie naar euro gemiddelde koerswaarde per jaar',
              fontsize=15, y=1.06)
ax3.set_ylabel('Aantal roepie')
ax3.set_xlabel('Jaar')

labels = ['Data Scientist','Data Engineer','Data Analyst', 'Machine Learning Engineer', 'Overige 44 beroepen']
sizes = [143, 132, 97, 41, 175]
total = len(data.Designation)
def fmt(x):
    return '{:.1f}%\n{:.0f}'.format(x, total*x/100)
piechartdiscussie, ax4 = plt.subplots()
ax4.pie(sizes, labels = labels, autopct=fmt,wedgeprops = {'linewidth' : 3, 'edgecolor' : 'white' })


data_kasper=data_kasper[data_kasper.groupby('Designation')['Designation'].transform('count').ge(7)]
kasper = px.bar(data_frame=data_kasper.groupby(['Designation']).mean().reset_index(),
    x="Designation",
    y="Salary_eur",
    color='Designation',
    text_auto='.2s',
    labels={'Salary_eur':'Salaris in euro per jaar', 'Designation':'Beroep'},height=500,
    title="Salaris per jaar per beroep in de US")
kasper.add_hline(y=data_kasper['gdp_euro'].mean() , line_width=3, line_dash="dash", line_color="black")




