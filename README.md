# Titanic-ML

This is an analysis of titanic survivor data and a ML model that predicts passenger's survival rate with logistic linear regression.
Trained and tested with the data from [here](https://www.kaggle.com/c/titanic/data)

## Tech Stack
```
Language: Python
Frameworks: Pandas, Numpy, scikit-Learn
```
## Data Analysis
### Data frame looks like 
![Screen Shot 2021-07-10 at 1 08 05 PM](https://user-images.githubusercontent.com/66694451/125171020-f5734300-e17f-11eb-8065-ccf9ca10acf8.png)

### visualize the ratio between survived and not survived
![Screen Shot 2021-07-11 at 4 28 35 PM](https://user-images.githubusercontent.com/66694451/125209167-0d70c280-e265-11eb-934a-31d705e643db.png)

### Checking the "Sex" feature and the corresponding Survival rate
![Screen Shot 2021-07-11 at 4 29 18 PM](https://user-images.githubusercontent.com/66694451/125209176-27aaa080-e265-11eb-9100-44fe1ca15a8a.png)

### relationship between Pclass and survival rate.
![Screen Shot 2021-07-11 at 4 30 57 PM](https://user-images.githubusercontent.com/66694451/125209268-a1db2500-e265-11eb-9aed-7e2400f2ac1e.png)
![Screen Shot 2021-07-11 at 4 31 34 PM](https://user-images.githubusercontent.com/66694451/125209279-b6b7b880-e265-11eb-994e-754923fd6b42.png)
We found that as people pay more money, the survival rate tends to be higher

### relationship between age and survival rate
![Screen Shot 2021-07-11 at 4 41 48 PM](https://user-images.githubusercontent.com/66694451/125209463-e87d4f00-e266-11eb-9f22-278a3bbbf5cb.png)

### heatmap
![Screen Shot 2021-07-11 at 4 44 27 PM](https://user-images.githubusercontent.com/66694451/125209523-4f026d00-e267-11eb-84ac-07ec7bc65712.png)

### Conclusion on analysis:
- "Sex" is one of the main predictor
- "Pclass" is one of the main predictor (the higher the Pclass, the higher the survival rate)
- Male with Pclass 3 has the least survival rate
- "Age" is not a effective predictor

## Feature engineering:

### Create a new column and have the entry as 1 wherever the column "Sex" shows "female" and 0 wherever it shows "male"
![Screen Shot 2021-07-08 at 10 54 50 AM](https://user-images.githubusercontent.com/66694451/124943962-f0c15a00-dfda-11eb-92f2-a5b2356bbd39.png)




