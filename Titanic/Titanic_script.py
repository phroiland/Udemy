'''Basic questions:

    1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
    2.) What deck were the passengers on and how does that relate to their class?
    3.) Where did the passengers come from?
    4.) Who was alone and who was with family?

Dig deeper, with a broader question:

    5.) What factors helped someone survive the sinking?'''

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as mat
import seaborn as sea
#% matplotlib inline

titanic_df = pd.read_csv('train.csv')

titanic_df.head()

titanic_df.info()

sexplot = sea.factorplot('Sex', data = titanic_df, kind = 'count')
sexplot.savefig('sexplot.png')

sexclass = sea.factorplot('Sex', data = titanic_df, kind = 'count', hue = 'Pclass')
sexclass.savefig('sexclass.png')

sexsurvive = sea.factorplot('Sex', data = titanic_df, kind = 'count', hue = 'Survived')
sexsurvive.savefig('sexsurvive.png')

pclass_sex = sea.factorplot('Pclass', data = titanic_df, kind = 'count', hue = 'Sex')
pclass_sex.savefig('pclass_sex.png')

pclass_survive = sea.factorplot('Pclass', data = titanic_df, kind = 'count', hue = 'Survived')
pclass_survive.savefig('pclass_survive.png')

#Original age
titanic_df['Age'].mean()

#Age 
fig, (axis1,axis2) = mat.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')
#get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()
#generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
#fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
#convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
#plot original Age values
#original_age = titanic_df['Age'].hist(bins=70, ax=axis1)
#original_age.savefig('original_age.png')
#plot adjusted Age Values
#adjusted_age = titanic_df['Age'].hist(bins=70, ax=axis2)
#adjusted_age.savefig('adjusted_age.png')

#Adjusted age
titanic_df['Age'].mean()

def male_female_child(passenger):
    sex, age = passenger
    if sex == 'female':
        if age < 16:
            return 1
        elif age >= 16:
            return 3
        else:
            return 3
    if sex == 'male':
        if age < 16:
            return 2
        elif age >= 16:
            return 4
        else:
            return 4

titanic_df['Person'] = titanic_df[['Sex', 'Age']].apply(male_female_child, axis=1)

person_pclass = sea.factorplot('Pclass', data = titanic_df, kind = 'count', hue = 'Person')
person_pclass.savefig('person_pclass.png')

titanic_df['Person'].value_counts()

fig = sea.FacetGrid(titanic_df, hue = 'Sex', aspect = 4)
fig.map(sea.kdeplot, 'Age', shade = True)
oldest = titanic_df['Age'].max()
fig.set = ((0, oldest))
fig.add_legend()
fig.savefig('sex_age.png')

fig2 = sea.FacetGrid(titanic_df, hue = 'Person', aspect = 4)
fig2.map(sea.kdeplot, 'Age', shade = True)
oldest = titanic_df['Age'].max()
fig2.set = ((0, oldest))
fig2.add_legend()
fig2.savefig('person_age.png')

fig3 = sea.FacetGrid(titanic_df, hue = 'Pclass', aspect = 4)
fig3.map(sea.kdeplot, 'Age', shade = True)
oldest = titanic_df['Age'].max()
fig3.set = ((0, oldest))
fig3.add_legend()
fig3.savefig('pclass_age.png')

#cabin column 
titanic_df['Deck'] = titanic_df['Cabin'].str[0]

#Only need first letter of cabin for deck
#levels = []
#for level in deck:
    #levels.append(level[0])
#cabin_df = DataFrame(levels)
#cabin_df.columns = ['Cabin']
summer_deck = sea.factorplot('Deck', data = titanic_df, palette = 'summer', kind = 'count')
summer_deck.savefig('summer_deck.png')

embarked_pclass = sea.factorplot('Embarked', data = titanic_df, hue = 'Pclass', x_order = ['C', 'Q', 'S'], kind = 'count')
embarked_pclass.savefig('embarked_pclass.png')

#Who was with family or traveling alone
titanic_df['Travel'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['Travel'].loc[titanic_df['Travel'] > 0] = 'Family'
titanic_df['Travel'].loc[titanic_df['Travel'] == 0] = 'Alone'

travel_count = sea.factorplot('Travel', data = titanic_df, kind = 'count', palette = 'ocean_d')
travel_count.savefig('travel_count.png')

#Tongue-in-cheek reference to the film Titanic--survivors are Rose, those that died are Jack
titanic_df['Outcome'] = titanic_df.Survived.map({0:'Jack', 1:'Rose'})

jack_rose1 = sea.factorplot('Outcome', data = titanic_df, palette = 'ocean_d', kind = 'count')
jack_rose1.savefig('jack_rose1.png')

fig4 = sea.FacetGrid(titanic_df, hue = 'Outcome', aspect = 4)
fig4.map(sea.kdeplot, 'Age', shade = True)
oldest = titanic_df['Age'].max()
fig4.set = ((0, oldest))
fig4.add_legend()
fig4.savefig('outcome_age.png')

survive1 = sea.factorplot('Pclass', 'Survived', hue = 'Person', data = titanic_df, palette = 'ocean')
survive1.savefig('survive1.png')

survive2 = sea.lmplot('Age', 'Survived', hue = 'Pclass', data = titanic_df, palette = 'ocean')
survive2.savefig('survive2.png')

survive3 = sea.lmplot('Age', 'Survived', hue = 'Sex', data = titanic_df, palette = 'summer')
survive3.savefig('survive3.png')

survive4 = sea.lmplot('Age', 'Survived', hue = 'Travel', data = titanic_df, palette = 'summer')
survive4.savefig('survive4.png')

jack_rose2 = sea.lmplot('Person', 'Age', hue = 'Outcome', data = titanic_df, palette = 'rainbow')
jack_rose2.savefig('jack_rose2.png')

jack_rose_all = sea.pairplot(titanic_df, hue = 'Outcome')
jack_rose_all.savefig('jack_rose_all.png')

embarked_all = sea.pairplot(titanic_df, hue = 'Embarked', palette = 'rainbow')
embarked_all.savefig('embarked_all.png')