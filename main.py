import ftfy
from streamlit import *
from data_backend import *
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np


titel = container()
introductie = container()
api_uitleg = container
data_analyse = container()
head = container()
feature_1 = container()
feature_2 = container()
dataset_2 = container()
gdp_merge = container()
feature_engineering = container()
visualisatie = container()
salary_pred = container()


with titel:
    title("Analyse van salarissen in de data science wereld")
    #image(intro_image)
with introductie:
    write(ftfy.fix_encoding(t_introductie))
    header('Data importeren via de kaggle API')
    write(api)
    code("""

#Initialisatie
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

#Dataset downloaden
!kaggle datasets download -d zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths
api.dataset_download_file('zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths',
                         file_name = 'gdp_per_capita.csv')
!kaggle datasets download -d zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths

#Uitpakken
import zipfile

with zipfile.ZipFile("data-science-fields-salary-categorization.zip","r") as file:
    file.extractall("Salaris")

with zipfile.ZipFile("world-gdpgdp-gdp-per-capita-and-annual-growths.zip","r") as file:
    file.extractall("Gdp")

  """, language='python')


with data_analyse:
    header('Data analyse')
    write(ftfy.fix_encoding(Dataanalyse))
    write(raw.head())
    subheader('Univariate data analyse')
    write(ftfy.fix_encoding(UnivariateData))
    tab1_uni, tab2_uni = tabs(['Blog', 'Code'])
    with tab1_uni:
        left_ana, right_ana=columns(2)
        with left_ana:
            list_choice = ['Working Year', 'Designation', 'Experience', 'Employment Status', 'Salary In Rupees', 'Employee Location', 'Company Location', 'Company Size', 'Remote Working Ratio', 'Count']
            choice_dict = {'Working Year': 'Working_Year', 'Designation': 'Designation', 'Experience': 'Experience', 'Employment Status': 'Employment_Status', 'Salary In Rupees': 'Salary_In_Rupees', 'Employee Location': 'Employee_Location', 'Company Location': 'Company_Location', 'Company Size': 'Company_Size', 'Remote Working Ratio': 'Remote_Working_Ratio', 'Count': 'Count'}

            feature_choice=selectbox('Selecteer de kolom', options=list_choice)

            feature_choice=choice_dict[feature_choice]
            bins=slider('Aantal bins', min_value=4, max_value=100)
        with right_ana:
            plot_choice=selectbox('Plot keuze', options=["box",'violin', 'rug'])
            coloryesno = checkbox('Colorscale')
        figureee = px.histogram(raw.dropna(), x=feature_choice,
                                marginal=plot_choice,  # or violin, rug
                                hover_data=raw.columns, nbins=bins, color=(feature_choice if coloryesno else None))
        plotly_chart(figureee)
    with tab2_uni:
        
        code("""# 2 kolommen maken
left_ana, right_ana=columns(2)
with left_ana: 
  feature_choice=selectbox('Selecteer de kolom', options=raw.columns)
  bins=slider('Aantal bins', min_value=4, max_value=100)
               
with right_ana:
  plot_choice=selectbox('Plot keuze', options=["box",'violin', 'rug'])
  coloryesno = checkbox('Colorscale')
                
figureee = px.histogram(raw.dropna(), x=feature_choice,
                                marginal=plot_choice,
                                hover_data=raw.columns, nbins=bins, 
                                color=(feature_choice if coloryesno else None))
plotly_chart(figureee)""", language='python')



with gdp_merge:
    header('Integreren van de BBP per hoofd data')
    write(ftfy.fix_encoding(wereldbol))
    tab1_uni, tab2_uni = tabs(['Blog', 'Code'])
    with tab1_uni:
        merge_l, merge_r = columns(2)
        with merge_l:
            option = radio('Selecteer visualisatie', options=['Kaart', 'Treemap'])
        if option == 'Kaart':
            with merge_r:
                kaart_soort = radio("Is de aarde plat?", options=['Ja', 'Nee'])
            if kaart_soort=='Ja':
                plotly_chart(gdp_map_plat, use_container_width=True)
            else:
                plotly_chart(gdp_map_bol, use_container_width=True)
        else:
            plotly_chart(gdp_treemap, use_container_width=True)
            
    with tab2_uni:
        ##Wereldbol
        code("""
    merge_l, merge_r = columns(2)
    with merge_l:
        option = radio('Selecteer visualisatie', options=['Kaart', 'Treemap'])
    if option == 'Kaart':
        with merge_r:
            kaart_soort = radio("Is de aarde plat?", options=['Ja', 'Nee'])
        if kaart_soort=='Ja':
            plotly_chart(gdp_map_plat, use_container_width=True)
        else:
            plotly_chart(gdp_map_bol, use_container_width=True)
    else:
        plotly_chart(gdp_treemap, use_container_width=True)
        """, language='python')        
        
with feature_engineering:
    header('Feature engineering')
    write(ftfy.fix_encoding(t_feature_1))
    f_left, f_right = columns(2)
    with f_left:
       pyplot(fig2, caption = 'Figuur x')
    with f_right:
        write(ftfy.fix_encoding(t_feature_2))
    write(ftfy.fix_encoding(t_feature_3))
    write(data.head())
    
    
    
with visualisatie:
    header('Visualisatie')
    write(ftfy.fix_encoding(Visualisatie))
    tab1_uni, tab2_uni = tabs(['Blog', 'Code'])
    
    with tab1_uni:
        aantal_jobs = slider('Aantal beroepen', min_value=0, max_value=50, value=25)

        kasper = px.bar(data_frame=final_kasper.head(aantal_jobs),
                        x="Designation",
                        y="Salary_eur",
                        color='Designation',
                        text_auto='.2s',
                        labels={'Salary_eur': 'Salaris in euro per jaar', 'Designation': 'Beroep'}, height=500,
                        title="Salaris per jaar per beroep in de US")
        kasper.add_hline(y=data_kasper['gdp_euro'].mean(), line_width=3, line_dash="dash", line_color="black")
        plotly_chart(kasper,use_container_width=True)

        scatterplot_salaris, ax = plt.subplots()
    with tab2_uni:
        #Visualisatie
        code("""
        with visualisatie:
    aantal_jobs = slider('Aantal beroepen', min_value=0, max_value=50, value=25)

    kasper = px.bar(data_frame=final_kasper.head(aantal_jobs),
                    x="Designation",
                    y="Salary_eur",
                    color='Designation',
                    text_auto='.2s',
                    labels={'Salary_eur': 'Salaris in euro per jaar', 'Designation': 'Beroep'}, height=500,
                    title="Salaris per jaar per beroep in de US")
    kasper.add_hline(y=data_kasper['gdp_euro'].mean(), line_width=3, line_dash="dash", line_color="black")
    plotly_chart(kasper,use_container_width=True)
        """, language='python')
    
    
    begin, checks, isernogniet, code_vis = tabs(['Home', 'Subset', 'Kleur', 'Code'])
    with checks:
        f,g, h=columns(3)
        with g:
            markdown('Bedrijfsgrootte')
            Klein=checkbox('Klein', value = True)
            Medium=checkbox('Medium', value = True)
            Groot=checkbox('Groot', value = True)
        with h:
            markdown('Jaar')
            j2020=checkbox('2020', value = True)
            j2021=checkbox('2021', value = True)
            j2022=checkbox('2022', value = True)
        with f:
            markdown('Ervaring')
            Starter = checkbox('Starter', value = True)
            Medior = checkbox('Medior', value = True)
            SeniorExpert = checkbox('Senior/Expert', value = True)
            Executive=checkbox('Executive', value = True)

    subdata = data

    if not Klein:
        subdata = subdata[subdata['Grootte'] != 'S']
    if not Medium:
        subdata = subdata[subdata['Grootte'] != 'M']
    if not Groot:
        subdata = subdata[subdata['Grootte'] != 'L']
    if not j2020:
        subdata = subdata[subdata['Jaar'] != 2020]
    if not j2021:
        subdata = subdata[subdata['Jaar'] != 2021]
    if not Starter:
        subdata = subdata[subdata['Experience'] != 'EN']
    if not Medior:
        subdata = subdata[subdata['Experience'] != 'MI']
    if not SeniorExpert:
        subdata = subdata[subdata['Experience'] != 'SE']
    if not Executive:
        subdata = subdata[subdata['Experience'] != 'EX']

    x_scatter=subdata['GDP €']+np.random.normal(0,5000,len(subdata['GDP €']))
    y_scatter=subdata['Salaris €']


    #pyplot(scatterplot_salaris)
    with begin:    
        k,l=columns(2)
        scatterplotly_x = k.selectbox('x-as', options=subdata.columns)
        scatterplotly_y = l.selectbox('y-as', options=subdata.columns)

        try:
            distortion_x = float(k.text_input('Ruis-x', value='0'))
            distortion_y = float(l.text_input('Ruis-y', value='0'))
        except:
            raise ValueError('Het moet een getal zijn')
        opacity = k.slider('Doorzichtigheid', max_value=1000, min_value=0, value=1000, step=1)/1000
        trendline = l.selectbox('Trendlijn', options=[None,'ols', 'lowess',  'expanding' ])
    tab1_uni, tab2_uni = tabs(['Blog', 'Code'])
    
    
    with isernogniet:
        isernogniet_left, isernogniet_right = columns(2)
        
        with isernogniet_left:
            kleur_keuze_options = [None, 'Jaar', 'Designation', 'Experience', "Land persoon", 'Land bedrijf', 'expat', 
                                   'Grootte', 'Thuiswerk ratio']
            kleur_keuze = selectbox('Selecteer categorie voor kleuren', options=kleur_keuze_options)
    
    with tab1_uni:                
        try:
            scatterplotly = px.scatter(subdata,
                                               x=subdata[scatterplotly_x]+np.random.normal(0,distortion_x,len(subdata)),
                                               y=subdata[scatterplotly_y]+np.random.normal(0,distortion_y,len(subdata)),
                                               labels={'x':str(scatterplotly_x), 'y':str(scatterplotly_y)},
                                               trendline=trendline,
                                               opacity=opacity,
                                               trendline_color_override='red', color=kleur_keuze
                                               )
            plotly_chart(scatterplotly)
        except:
            write('Verander parameters')
        try:
            scatterplotly = px.bar(x=subdata[scatterplotly_x],
                                               y=subdata[scatterplotly_y],
                                               labels={'x': str(scatterplotly_x), 'y': str(scatterplotly_y)},
                                               opacity=opacity)
            plotly_chart(scatterplotly)
            write(subdata.columns)
        except:
            write('Verander parameters')
    with tab2_uni:
        code("""
        try:
            scatterplotly = px.scatter(x=subdata[scatterplotly_x]+np.random.normal(0,distortion_x,len(subdata)),
                                               y=subdata[scatterplotly_y]+np.random.normal(0,distortion_y,len(subdata)),
                                               labels={'x':str(scatterplotly_x), 'y':str(scatterplotly_y)},
                                               trendline=trendline,
                                               opacity=opacity,
                                               trendline_color_override='red')
            plotly_chart(scatterplotly)
        except:
            scatterplotly = px.bar(x=subdata[scatterplotly_x],
                                               y=subdata[scatterplotly_y],
                                               labels={'x': str(scatterplotly_x), 'y': str(scatterplotly_y)},
                                               opacity=opacity)
            plotly_chart(scatterplotly)""", language='python')
    

with salary_pred:
    header('De salaris voorspeller')
    #image(salary_image)
    write('Benieuwd wat je gaat vedienen als je gaat werken in de data wereld? Stel jouw '
          'toekomstige baan samen door de '
          'onderstaande mini enquete in te vullen en '
          'kom er achter was je salaris gaat worden!')

    
    tab1_uni, tab2_uni = tabs(['Blog', 'Code'])
    with tab1_uni:
        des = selectbox('Selecteer beroep', options=data.Designation.unique())
        pred_left, pred_right = columns(2)
        with pred_left:
            ex = select_slider('Selecteer ervaring', options=['Entry', 'Medior', 'Senior/Expert', 'Executive'])
            size = selectbox('Grootte bedrijf', options=['Groot', 'Medium', 'Klein'])
            la_pe = selectbox('Land persoon', options=data['Land persoon'].unique())
        with pred_right:
            rat = slider('Thuiswerk ratio', min_value=0, max_value=100, step=50)
            stat = selectbox('Selecteer dienstverband', options=['Fulltime', 'Parttime', 'Oproepbasis', 'Freelance'])
            la_be = selectbox('Land bedrijf', options=data['Land bedrijf'].unique())
    

    size = size_dict[size]
    stat = stat_dict[stat]
    ex = ex_dict[ex]
    if button('Genereer'):
        percent, salarisS = salary_prediction(des, ex, stat, la_pe, la_be, size, rat)
        subheader(f'Het voorspelde salaris is {salarisS}, dat is {percent}% ten opzichte van het gdp')
    
    
    
    
    with tab2_uni:
        ##Salaris voorspeller
        code("""
with salary_pred:
    des = selectbox('Selecteer beroep', options=data.Designation.unique())
    pred_left, pred_right = columns(2)
    with pred_left:
        ex = select_slider('Selecteer ervaring', options=['Entry', 'Medior', 'Senior/Expert', 'Executive'])
        size = selectbox('Grootte bedrijf', options=['Groot', 'Medium', 'Klein'])
        la_pe = selectbox('Land persoon', options=data['Land persoon'].unique())
    with pred_right:
        rat = slider('Thuiswerk ratio', min_value=0, max_value=100, step=50)
        stat = selectbox('Selecteer dienstverband', options=['Fulltime', 'Parttime', 'Oproepbasis', 'Freelance'])
        la_be = selectbox('Land bedrijf', options=data['Land bedrijf'].unique())

    size = size_dict[size]
    stat = stat_dict[stat]
    ex = ex_dict[ex]
    if button('Genereer'):
        percent, salarisS = salary_prediction(des, ex, stat, la_pe, la_be, size, rat)
        subheader(f'Het voorspelde salaris is {salarisS}, dat is {percent}% ten opzichte van het gdp')
        """, language='python')

    
    
        
        
    
    header('Discussie')
    #fgf, gfg = columns(2)
    #with fgf:
   # write("Piechart verdeling beroepen")
    
    #with gfg:
    write(Discussie)
    image(appeltaart)
    write(Discussie2)
    image(qrcode)

    text("Deze website is gemaakt door: Daan van der Hoek, Kasper Goedings, Kevin Kosters, ")
    text('Brandon Haak')
    
    
    
    #pyplot(piechartdiscussie)
