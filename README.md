# Titanic-ML

This is an analysis of titanic survivor data and a ML model that predicts passenger's survival rate with logistic linear regression.
Trained and tested with the data from [Kaggle](https://www.kaggle.com/c/titanic/data)

## Tech Stack
```
Language: Python
Frameworks: Pandas, Numpy, scikit-Learn
```
## Data Analysis
### Data frame looks like 
![Screen Shot 2021-07-10 at 1 08 05 PM](https://user-images.githubusercontent.com/66694451/125171020-f5734300-e17f-11eb-8065-ccf9ca10acf8.png)

### Visualize the ratio between survived and not survived
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
### Create new features "male","female","C","Q","S"(which are from "Embarked" feature originally)
![Screen Shot 2021-07-11 at 6 33 31 PM](https://user-images.githubusercontent.com/66694451/125211747-8298c380-e276-11eb-9f89-366de2f8ebd5.png)

### Filled the null value with method of backfill and drop the irrelevent features
![Screen Shot 2021-07-11 at 6 34 47 PM](https://user-images.githubusercontent.com/66694451/125211768-b1169e80-e276-11eb-9129-36c454a22c78.png)

### Classification report(86% accuracy)
![Screen Shot 2021-07-11 at 6 40 14 PM](https://user-images.githubusercontent.com/66694451/125211851-7bbe8080-e277-11eb-86f7-ce1a35a1ee29.png)





