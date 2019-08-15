##There are two options ...
#1.train your own data and get whatever output you want
#1.directly get output from the by simply giving input


import click

@click.command()
#@click.option('--string',default='sudarshan')



def train():
            """Train your data here"""
              
            import pandas as pd
            ##applying label to the data
            
            file1=input("Enter first dataset to be trained")
            data1=pd.read_csv(file1)
            #only taking X,Y,Z,RGB VALUES for training
            data1=data1[['X','Y','Z','Red','Green','Blue']]
            #adding a new column named 'class ' to the dataset and putting its value as zero for ground
            data1['class']=0
            print(data1)
		    
            file2=input("Enter second dataset to be trained")
            data2=pd.read_csv(file2)
            data2=data2[['X','Y','Z','Red','Green','Blue']]
            #adding a new column named 'class ' to the dataset and puttong its value as one for nonground
            data2['class']=1
            print(data2)

            #concatting both datasets
            data3 = pd.concat([data1, data2], ignore_index=True)
            print(data3)
            data3
            #for applying logistic regression taking x and y values
            X=data3.drop("class",axis=1)
            y=data3['class']
            
            #from sklearn import necessary modules
            from sklearn.model_selection import train_test_split
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
            #implementing logistic regression
            from sklearn.linear_model import LogisticRegression
            logmodel=LogisticRegression(solver='lbfgs')
            logmodel.fit(X_train,y_train)
            prediction=logmodel.predict(X_test)

            ##classification report
            from sklearn.metrics import classification_report
            print(classification_report(y_test,prediction))
            ##confusion matrix
            from sklearn.metrics import confusion_matrix
            print('Confusion matrix:\n')
            print(confusion_matrix(y_test,prediction))
            ## accuracy score
            from sklearn.metrics import accuracy_score
            print('Accuracy score:  ')
            print(accuracy_score(y_test,prediction))

            #entering new data
            print('........................................................')
            file3=input("Enter your file name with location:.......")
            data4=pd.read_csv(file3)
            data4=data4[['X','Y','Z','Red','Green','Blue']]
 
            logmodel=LogisticRegression()
            logmodel.fit(X_train,y_train)
             
            #predictiong output for the entered data
            prediction=logmodel.predict(data4)
            cluster_map = pd.DataFrame()
            cluster_map['index']=data4.index.values
            cluster_map['class1']=prediction
            print('...........................................................\n')
            print('Choose from below \n')
            print('Ground: 1')
            print('Non ground: 2')
            choose1=input("Enter what do you want?")
            #if predicted output is one then put the in the ground output file..
            if choose1==1:
               a=cluster_map[cluster_map.class1==0]
               data4=pd.read_csv(file3)
               gnd=data4.iloc[a.index]
   
               file4=input('Enter the file name you want to write into...')
               gnd.to_csv(file4)
            elif choose1==2:
               b=cluster_map[cluster_map.class1==1]

               data4=pd.read_csv(file3)
               ngnd=data4.iloc[b.index]

               file4=input('Enter the file name you want to write into...')
               ngnd.to_csv(file4)
            else:
               print("choose a valid option....")


def non_ground():
    import pdal
    import pandas as pd
    import numpy as np
    in_file=input("Enter location and name of your file:.")
   
    data=pd.read_csv(in_file)
    data=data[['X','Y','Z','Red','Green','Blue']]

    data.insert(loc=0,column='A',value=1)
    #if rgb values are 8 bit or 16 bit
    if data['Red'].max()>255:

          a=[5.71504254e-12,-4.27504360e-06,  9.89084203e-07,  1.24089466e-06,
            -1.88929235e-04, -4.21298779e-07,  1.82733218e-04]
    else:
          a=[-3.84030028e-09,-7.54867545e-05,  5.41120395e-05,  9.33515601e-04,
            -2.46065068e-04,  1.16586638e-04,  1.49775974e-04] 
    b=np.asarray(a)
    data=data.mul(b.transpose())

    data['B']=data.sum(axis=1)

    new=pd.DataFrame()
    new['index']=data.index.values
    new['classes']=data.B
    new
    a=new[new.classes>=0]
    a
    data=pd.read_csv(in_file)
    ground=data.iloc[a.index]
    
    out_file=input("Enter the file location where you want to store non-ground points:..")
    ground.to_csv(out_file)
    




def ground():
    import pandas as pd
    import numpy as np
    in_file=input("Enter location and name of your file:.")
    data=pd.read_csv(in_file)
    data=data[['X','Y','Z','Red','Green','Blue']]



    #adding a new column so that we can multiply below array to each row of the dataset
    data.insert(loc=0,column='A',value=1)
    if data['Red'].max()>255:

          a=[5.71504254e-12,-4.27504360e-06,  9.89084203e-07,  1.24089466e-06,
            -1.88929235e-04, -4.21298779e-07,  1.82733218e-04]
    else:
          a=[-3.84030028e-09,-7.54867545e-05,  5.41120395e-05,  9.33515601e-04,
            -2.46065068e-04,  1.16586638e-04,  1.49775974e-04] 
    b=np.asarray(a)
    #muliplying each row with the array of co-efficients
    data=data.mul(b.transpose())

    #adding entirs of each row
    data['B']=data.sum(axis=1)

    #creating new dataframe for comparision purpose
    new=pd.DataFrame()
    new['index']=data.index.values
    new['classes']=data.B
   
    a=new[new.classes<0]
    
    data=pd.read_csv(in_file)
    ground=data.iloc[a.index]
    ground
    out_file=input("Enter the file location where you want to store ground points:..")
    ground.to_csv(out_file)
    print("File created!!")
    






	
