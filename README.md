# Titanic-ML

This a ML model that predicts passenger's survival rate given the information of age, fare paid or gender.
Trained and tested with the data from [here](https://www.kaggle.com/c/titanic/data)

## Tech Stack
```
Language: Python
Frameworks: Pandas, Numpy, scikit-Learn
```
### Data frame looks like 
![Screen Shot 2021-07-10 at 1 08 05 PM](https://user-images.githubusercontent.com/66694451/125171020-f5734300-e17f-11eb-8065-ccf9ca10acf8.png)

### Create a new column and have the entry as 1 wherever the column "Sex" shows "female" and 0 wherever it shows "male"
![Screen Shot 2021-07-08 at 10 54 50 AM](https://user-images.githubusercontent.com/66694451/124943962-f0c15a00-dfda-11eb-92f2-a5b2356bbd39.png)

### Age and Fare value normalized on a scale of {0,1}
![Screen Shot 2021-07-08 at 10 56 39 AM](https://user-images.githubusercontent.com/66694451/124944342-3bdb6d00-dfdb-11eb-8612-c6fdf3afcc83.png)

### selected train and test date set at a ratio of 8:2
![Screen Shot 2021-07-10 at 3 14 10 PM](https://user-images.githubusercontent.com/66694451/125174182-8272c800-e191-11eb-85e0-9f4c475be2df.png)
![Screen Shot 2021-07-10 at 3 14 15 PM](https://user-images.githubusercontent.com/66694451/125174183-843c8b80-e191-11eb-8711-33992f4d8ad3.png)

### Logistic Linear Regression using Gradient Descent, achieved an accuracy of 78%
![Screen Shot 2021-07-10 at 3 15 18 PM](https://user-images.githubusercontent.com/66694451/125174213-b221d000-e191-11eb-8dad-8b022c77fa94.png)


