import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
####
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


st.set_page_config(layout='wide')

## Data cleaning

df = pd.read_csv('export_dataframe.csv')

df[' school.mooe '] = df[' school.mooe '].str.replace(',', '').astype(float)
df = df[df['school.classification'] == 'Elementary']

df_1 = df[['School ID', 'Kinder Male', 'Kinder Female', 'Grade 1 Male',
       'Grade 1 Female', 'Grade 2 Male', 'Grade 2 Female', 'Grade 3 Male',
       'Grade 3 Female', 'Grade 4 Male', 'Grade 4 Female', 'Grade 5 Male',
       'Grade 5 Female', 'Grade 6 Male', 'Grade 6 Female', 'SPED NG Male',
       'SPED NG Female']]

# total number of student
df_1['total_stud'] = df_1['Kinder Male'] + df_1['Kinder Female'] + df_1['Grade 1 Male'] + df_1['Grade 1 Female'] + df_1['Grade 2 Male'] + df_1['Grade 2 Female'] + df_1['Grade 3 Male'] + df_1['Grade 3 Female'] + df_1['Grade 4 Male'] + df_1['Grade 4 Female'] + df_1['Grade 5 Male'] + df_1['Grade 5 Female'] + df_1['Grade 6 Male'] + df_1['Grade 6 Female'] + df_1['SPED NG Male'] + df_1['SPED NG Female']
df_tot_stud = df_1['total_stud']

# total number of teachers 
df_2 = df[['School ID','teachers.instructor','teachers.mobile','teachers.regular','teachers.sped']]
df_2['total_inst'] = (df_2['teachers.instructor'] + df_2['teachers.mobile'] + df_2['teachers.regular'] + df_2['teachers.sped'])
df_total_inst = df_2['total_inst']

# total number of rooms
df_3 = df[['rooms.standard.academic','rooms.standard.unused','rooms.nonstandard.academic','rooms.nonstandard.unused']]
df_3["rooms_total"] = (df['rooms.standard.academic'] +  df['rooms.standard.unused'] + df['rooms.nonstandard.academic'] +df['rooms.nonstandard.unused'])
df_rooms_total = df_3['rooms_total']

# school region and mooe
df_4 = df[['school.region',' school.mooe ','school.urban']]

final_df = pd.concat([df_tot_stud, df_total_inst, df_rooms_total, df_4], axis = 1)
final_df = final_df.dropna()

Q1 = final_df[' school.mooe '].quantile(0.25)
Q3 = final_df[' school.mooe '].quantile(0.75)
IQR = Q3 - Q1
final_df = (final_df[(final_df[' school.mooe '] >= Q1 - 1.5*IQR) & 
                           (final_df[' school.mooe '] <= Q3 + 1.5*IQR)])

final_df = final_df.dropna()
final_df = final_df.reset_index()


####
model_df = final_df.copy()

urb_labels = {'Partially Urban': 1, 'Urban': 0, 'Rural': 2}
model_df['school.urban'] = model_df['school.urban'].map(urb_labels)
model_df_1 = model_df[['total_stud','total_inst','rooms_total',' school.mooe ','school.urban']]

####

st.sidebar.title('Navigation Page')
navigation = st.sidebar.radio('Selection', ['Introduction', 'Concept Question and Methodology','Data Cleaning, EDA and Feature Engineering','Modelling','Result','Conclusion', 'Members'])
st.sidebar.write('------')
if navigation == 'Introduction':
    st.title('Unsupervised Machine Learning project using K-Means Clustering and DBSCAN on DepEd public school data')
    st.write('------')
    from PIL import Image
    img = Image.open('intropic.png')
    st.image(img, width = 800)
    st.write('------')

if navigation == 'Concept Question and Methodology':
    st.title('Unsupervised Machine Learning project using K-Means Clustering and DBSCAN on DepEd public school data')
    st.write('------')
    st.header('Concept Question')
    st.write('* Is there a difference with the actual MOOE of schools vs expected MOOEs computed by DepEd?')
    st.write('* What are the factors affecting the difference?')
    st.write('------')

    st.header('Methodology')
    st.subheader('Data Cleaning')
    st.write('* All null and duplicates was dropped.')
    st.write('* Outliers were also filter.')
    st.write('* All non-numeric values were manually encoded.')
    st.write('* PCA was used for preprocessing and reduced to 2 dimensions only (to help with clustering visualization)')
    st.write('* MOOE Eq. = School’s MOOE = P40000 + (P3000 x Number of Classrooms) + (P4000 x Number of Teachers) + (P200 x Number of Learners)')
    st.write('------')
          
if navigation == 'Data Cleaning, EDA and Feature Engineering':
    st.title('Unsupervised Machine Learning project using K-Means Clustering and DBSCAN on DepEd public school data')
    st.write('------')
    st.header('Data Cleaning, EDA and Feature Engineering')
    st.write('------')
    st.subheader('Preview Raw Data Frame')
    method_radio = st.radio('', ['Head','Tail','Shape'])
    if method_radio == 'Head':
        head_level = st.slider('Number of Rows: ', 10,50)
        st.write(df.head(head_level))
        st.success('Loading Successful!')
        st.write('------')
    if method_radio == 'Tail':
        tail_level = st.slider('Number of Rows: ', 10,50)
        st.write(df.tail(tail_level))
        st.success('Loading Successful!')
        st.write('------')
    if method_radio == 'Shape':
        if st.checkbox('Number of Rows'):
            st.write(len(df))
        if st.checkbox('Number of Columns'):
            st.write(len(df.columns))
            st.write(df.columns)
            st.write('------')
            
    st.header('Encoded Data Frame')
    st.write('The following features will be used for the modelling')
    st.write('* Total Number of Students')
    st.write('* Total Number of Teachers')
    st.write('* Total Number of Rooms')
    st.write('* School MOOE')
    st.write('* school urban')

    st.write('Legend: 0 - Urban, 1 - Partially Urban, 2 - Rural')
    st.dataframe(model_df.head(30)) 
    st.success('Loading Successful!')
    st.write('------')

if navigation == 'Modelling':
    st.title('Unsupervised Machine Learning project using K-Means Clustering and DBSCAN on DepEd public school data')
    st.write('------')
    st.header('Modelling')
    st.write('------')
    
    model_df_1 = model_df[['total_stud','total_inst','rooms_total',' school.mooe ','school.urban']]
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(model_df_1)
    model_df_pca = PCA(n_components = 2).fit_transform(scaled_df)
    
    kmodel1 = KMeans(n_clusters = 4, random_state = 42, init = 'random').fit(model_df_pca)
    pred_label_1 = kmodel1.predict(model_df_pca)

    st.subheader('Preview PCA Data Frame')
    st.subheader('PCA Data Frame')
    st.write(model_df_pca[:10])
    st.success('Loading Successful!')
    
    
    if st.sidebar.checkbox('Using KMeans Model'):
        st.header('KMean Options')
        cluster_slider = st.slider('Number of Cluster', 2,5)
        rs_slider = st.slider('Random State', 1, 100)

        kmodel1 = KMeans(n_clusters = cluster_slider, random_state = rs_slider).fit(model_df_pca)
        pred_label_1 = kmodel1.predict(model_df_pca)
        pred_label_df = pd.DataFrame(pred_label_1, columns = ['labels'])
        model_df_pca_df = pd.DataFrame(model_df_pca, columns = ['PCA1', 'PCA2'])
        gr_model = pd.concat([model_df_pca_df, pred_label_df], axis = 1)

        st.subheader('Inertia: ')
        st.write(kmodel1.inertia_)
        st.subheader('Silhouette Score:')
        st.write(silhouette_score(model_df_pca, pred_label_1))

        clusterp_0 = gr_model[gr_model['labels'] == 0]
        clusterp_1 = gr_model[gr_model['labels'] == 1]
        clusterp_2 = gr_model[gr_model['labels'] == 2]
        clusterp_3 = gr_model[gr_model['labels'] == 3]
        clusterp_4 = gr_model[gr_model['labels'] == 4]

        fig1 = plt.figure(figsize = (9,9))
        sns.scatterplot(clusterp_0['PCA1'], clusterp_0['PCA2'])
        sns.scatterplot(clusterp_1['PCA1'], clusterp_1['PCA2'])
        sns.scatterplot(clusterp_2['PCA1'], clusterp_2['PCA2'])
        sns.scatterplot(clusterp_3['PCA1'], clusterp_3['PCA2'])
        sns.scatterplot(clusterp_4['PCA1'], clusterp_4['PCA2'])
        st.pyplot(fig1)
        st.success('Loading Successful!')

        # labelled data
        st.write(gr_model[:50])
               
    if st.sidebar.checkbox('Using DBSCAN Model'):
        st.header('DBSCAN Model')
        st.subheader('Using only the Default Hyperparameters')
        
        kmodel1 = DBSCAN(min_samples = 3).fit(model_df_pca)
        pred_label_1 = kmodel1.labels_
        pred_label_df = pd.DataFrame(pred_label_1, columns = ['labels'])
        model_df_pca_df = pd.DataFrame(model_df_pca, columns = ['PCA1', 'PCA2'])
        gr_model = pd.concat([model_df_pca_df, pred_label_df], axis = 1)

        st.subheader('Silhouette Score:')
        st.write(silhouette_score(model_df_pca, pred_label_1))


        clusterp_0 = gr_model[gr_model['labels'] == 0]
        clusterp_1 = gr_model[gr_model['labels'] == 1]
        clusterp_2 = gr_model[gr_model['labels'] == 2]
        clusterp_3 = gr_model[gr_model['labels'] == 3]
        clusterp_4 = gr_model[gr_model['labels'] == 4]

        fig1 = plt.figure(figsize = (7,7))
        sns.scatterplot(clusterp_0['PCA1'], clusterp_0['PCA2'])
        sns.scatterplot(clusterp_1['PCA1'], clusterp_1['PCA2'])
        sns.scatterplot(clusterp_2['PCA1'], clusterp_2['PCA2'])
        sns.scatterplot(clusterp_3['PCA1'], clusterp_3['PCA2'])
        sns.scatterplot(clusterp_4['PCA1'], clusterp_4['PCA2'])
        st.pyplot(fig1)
        st.success('Loading Successful!')

        st.write(gr_model[:10])
        st.success('Loading Successful!')
        st.write(gr_model['labels'].value_counts())


if navigation == 'Result':
    st.title('Unsupervised Machine Learning project using K-Means Clustering and DBSCAN on DepEd public school data')
    st.write('------')
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(model_df_1)
    model_df_pca = PCA(n_components = 2).fit_transform(scaled_df)

    kmodel1 = DBSCAN(min_samples = 3).fit(model_df_pca)
    pred_label_1 = kmodel1.labels_
    pred_label_df = pd.DataFrame(pred_label_1, columns = ['labels'])
    model_df_pca_df = pd.DataFrame(model_df_pca, columns = ['PCA1', 'PCA2'])

    labeled_df = pd.concat([model_df_1, pred_label_df], axis = 'columns')
    labeled_df['comp.mooe'] = 3000*labeled_df['rooms_total'] + 4000*labeled_df['total_inst'] + 200*labeled_df['total_stud'] + 40000
    labeled_df['perc_diff'] = (labeled_df[' school.mooe '] - labeled_df['comp.mooe'])/labeled_df['comp.mooe'] * 100

    st.header('Results')
    st.write('------')

    # distribution
    # cluster 0
    cluster_0 = labeled_df[labeled_df['labels'] == 0]
    st.header('Cluster 0: Partially Urban')
    st.write('Number of schools: {}'.format(len(cluster_0)))
    fig_1 = plt.figure(figsize = (7,7))
    sns.distplot(cluster_0['perc_diff'])
    st.pyplot(fig_1)
    st.write('------')

    # cluster 1
    cluster_1 = labeled_df[labeled_df['labels'] == 1]
    st.header('Cluster 1:  Urban')
    st.write('Number of schools: {}'.format(len(cluster_1)))
    fig_2 = plt.figure(figsize = (7,7))
    sns.distplot(cluster_1['perc_diff'])
    st.pyplot(fig_2)
    st.write('------')


    cluster_2 = labeled_df[labeled_df['labels'] == 2]
    st.header('Cluster 2: Rural')
    st.write('Number of schools: {}'.format(len(cluster_2)))
    fig_3 = plt.figure(figsize = (7,7))
    sns.distplot(cluster_2['perc_diff'])
    st.pyplot(fig_3)
    st.write('------')
        
if navigation == 'Conclusion':
    st.title('Unsupervised Machine Learning project using K-Means Clustering and DBSCAN on DepEd public school data')
    st.write('------')
    st.header('Conclusion')
    st.write("Most of the schools' actual MOOEs are either less than or more than the MOOEs expected by the Department of Education. Overall, schools' actual MOOE is more than 11.83% than the expected MOOE. Thus, schools tend to spend more than what DepEd is expecting.")
    st.write('------')

if navigation == 'Members':
    st.title('Unsupervised Machine Learning project using K-Means Clustering and DBSCAN on DepEd public school data')
    st.write('------')
    st.header('Members')
    st.write('* Rods')
    st.write('* Razel')
    st.write('* Jay')
    st.write('* King')
    st.header('Mentor')
    st.write('* Ric Alindayu')

































