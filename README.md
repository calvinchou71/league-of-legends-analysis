# Prediction of the outcomes of League of Legends Matches after 10 minutes have passed.

by Calvin Chou (calchou@umich.edu)

## Introduction

League of Legends is a video game, where the ultimate goal is to be on the winning team. This Portfolio aims to accurately predict the outcome of League of Legends matches as this is the most important variable in the game. The dataset used is from 2022 League of Legends matches for various leagues. It contains information on a number different variables from the game. There are 150180 rows and 161 columns. Given we want to predict the outcome of match, certain variables are more relevant to our prediction than others. 

Our model will predict the result of a league of legends after 10 minutes of the match have passed. The most important feature columns are goldat10, xpat10, csat10, opp_goldat10, opp_xpat10, opp_csat10, golddiffat10, xpdiffat10, csdiffat10, killsat10, assistsat10, deathsat10, opp_killsat10, opp_assistsat10, and opp_deathat10. 

- goldat10 is how much gold a player has at the 10 minute mark. Gold is used to buy items to make your character stronger. 

- xpat10 is how much xp a player has at the 10 minute mark. XP is used to level up your character, which allows you to improve your abilities and makes your character stronger.

- csat10 is how many minions/monsters (cs) a player has killed at the 10 minute mark. cs is similiar to the amount of gold as getting cs gives you gold.

- opp_goldat10, opp_xpat10, opp_csat10 is the amount of gold, xp, and cs the opponent has at the 10 minute mark.

*Your opponent is the person on the other team in the same position as you. There are 5 roles in the game, top, jungle, mid, adc, and support.*

- golddiffat10, xpdiffat10, csdiffat10 is the difference between you and your opponent.

- killsat10 is the number of kills you have at 10 minutes.

- assistsat10 is the number of assists you have at 10 minutes. Assists are when you don't kill a player on the opposing team, but help in the kill.

- deathsat10 is the number of deaths you have at 10 minutes.

- opp_killsat10, opp_assistsat10, and opp_deathat10 are the number of kills, assists, and deaths for your opponent.


All of these columns reveal information about the state of the game at the 10 minute mark in the game. 

---

## Cleaning and EDA

### Cleaning

We seperated the dataset into a datasets. One where the dataset is filtered to contain rows for the entire team rather than a single player. This is useful when we predict the outcomes for an entire team. We then created a dataset for only the players to predict the outcomes for a single player.

We then fitlered the rows where the datacompletness column is equal to complete as the rows that aren't complete didn't were missing the values for our columns of interests. \\


Team Dataframe

```py
print(df_clean_team[['goldat10', 'xpat10', 'csat10', 'opp_goldat10', 'opp_xpat10', 'opp_csat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10', 'killsat10', 'assistsat10', 'deathsat10', 'opp_killsat10', 'opp_assistsat10', 'opp_deathsat10']].head().to_markdown(index=False))
```

|   goldat10 |   xpat10 |   csat10 |   opp_goldat10 |   opp_xpat10 |   opp_csat10 |   golddiffat10 |   xpdiffat10 |   csdiffat10 |   killsat10 |   assistsat10 |   deathsat10 |   opp_killsat10 |   opp_assistsat10 |   opp_deathsat10 |
|-----------:|---------:|---------:|---------------:|-------------:|-------------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|----------------:|------------------:|-----------------:|
|      16218 |    18213 |      322 |          14695 |        18076 |          330 |           1523 |          137 |           -8 |           3 |             5 |            0 |               0 |                 0 |                3 |
|      14695 |    18076 |      330 |          16218 |        18213 |          322 |          -1523 |         -137 |            8 |           0 |             0 |            3 |               3 |                 5 |                0 |
|      14939 |    17462 |      317 |          16558 |        19048 |          344 |          -1619 |        -1586 |          -27 |           1 |             1 |            3 |               3 |                 3 |                1 |
|      16558 |    19048 |      344 |          14939 |        17462 |          317 |           1619 |         1586 |           27 |           3 |             3 |            1 |               1 |                 1 |                3 |
|      15466 |    19600 |      368 |          15569 |        18787 |          355 |           -103 |          813 |           13 |           0 |             0 |            1 |               1 |                 1 |                0 |


Player Data Frame

```py
print(df_clean_player[['goldat10', 'xpat10', 'csat10', 'opp_goldat10', 'opp_xpat10', 'opp_csat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10', 'killsat10', 'assistsat10', 'deathsat10', 'opp_killsat10', 'opp_assistsat10', 'opp_deathsat10']].head().to_markdown(index=False))
```

|   goldat10 |   xpat10 |   csat10 |   opp_goldat10 |   opp_xpat10 |   opp_csat10 |   golddiffat10 |   xpdiffat10 |   csdiffat10 |   killsat10 |   assistsat10 |   deathsat10 |   opp_killsat10 |   opp_assistsat10 |   opp_deathsat10 |
|-----------:|---------:|---------:|---------------:|-------------:|-------------:|---------------:|-------------:|-------------:|------------:|--------------:|-------------:|----------------:|------------------:|-----------------:|
|       3228 |     4909 |       89 |           3176 |         4953 |           81 |             52 |          -44 |            8 |           0 |             0 |            0 |               0 |                 0 |                0 |
|       3429 |     3484 |       58 |           2944 |         3052 |           63 |            485 |          432 |           -5 |           1 |             2 |            0 |               0 |                 0 |                1 |
|       3283 |     4556 |       81 |           3121 |         4485 |           81 |            162 |           71 |            0 |           0 |             1 |            0 |               0 |                 0 |                1 |
|       3600 |     3103 |       78 |           3304 |         2838 |           90 |            296 |          265 |          -12 |           1 |             1 |            0 |               0 |                 0 |                0 |
|       2678 |     2161 |       16 |           2150 |         2748 |           15 |            528 |         -587 |            1 |           1 |             1 |            0 |               0 |                 0 |                1 |


We did not impute any values as we did not see a need to.


### EDA

<iframe
  src="assets/one_var.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/two_var.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/kills_bar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="assets/champ_win_bar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

```py
print(pivot_table.iloc[:, :10].to_markdown(index=False))
```

|    Aatrox |       Ahri |     Akali |   Akshan |   Alistar |   Amumu |   Anivia |   Annie |   Aphelios |       Ashe |
|----------:|-----------:|----------:|---------:|----------:|--------:|---------:|--------:|-----------:|-----------:|
| -709      |  429       | -679      | nan      |  nan      | nan     | nan      |  nan    |    62.4877 | -135.88    |
|  nan      | -315.5     |  nan      | nan      |  nan      | 218     | nan      |  nan    |   nan      |  nan       |
|  329.208  |    9.80447 |  -39.6323 | -18.0159 |  nan      | nan     |  75.7872 |  -30.25 |   411      |  nan       |
|  nan      |  nan       |  nan      | nan      |  -52.7232 | -18.061 | nan      |  nan    |   nan      |    9.77011 |
|   80.5134 |  594.833   |  -50.1492 | 199.283  |  nan      | nan     | nan      |  nan    |   268.5    |  nan       |

Only the first 10 champions are shown to save space.


## Framing a Prediction Problem

The variable we are trying to predict is the result column, which only has 2 possible values (1 and 0). 1 means the team won and 0 means the team lost. Therefore, the prediction problem is a binary classification problem. We chose the result as it is most important column as the winning is the objective of the game.

We want to predict the result after the 10 minute mark of the game. Given this we will have all match information that has occured within the first 10 minutes. We chose 10 minutes as it was the earliest point for predicting the result. We did not want to choose a later time as much of the result is already determined. We chose to use a logistic regression model as it works well on binary classification.

To test the efficacy of our model, we will use loss and accuracy. We chose loss as it works well on classification data because it penalizes confident misclassifications. We chose accuracy as the result column is perfectly balanced, so accuracy should work well.

## Baseline Model

For our baseline model, we used a logistic regression model on goldat10, xpat10, and csat10. All these features are quantitative and did not require any encodings.

The loss of the model was 0.6896490354686542 for players only.
The accuracy of the model was 0.5186914054838583 for players only.

The loss of the model was 0.6214919289200207 for teams only.
The accuracy of the model was 0.6552129912920687 for teams only.

This model did not perform that well. The loss for the players only model was 0.6896490354686542 and 0.6214919289200207 for the teams only model. These loss values are quite high.

The accuracy for the players only data was 0.5186914054838583 and the team only data was 0.6552129912920687. On a player by player basis our accuracy was essentially the same as random chance. We do expect the team only data to have a higher accuracy as it takes the aggreate of the entire team against the aggregate of the other team, so it should be more represenative of the game state. 

## Final Model

For our final model, we used a logistic regression model on golddiffat10, xpdiffat10, csdiffat10, killsat10, assistsat10, deathsat10. All these features are quantitative and did not require any encodings.

We added these features because it gives us more information on how the opponents are doing relative to the players as well as gives information around the kills, assists, and deaths, which are also very important to the game as being dead is time missed from completing actions.

We selected hyperparameters using a GridSearch.


The loss of the model was 0.6457005259987575 for players only.
The accuracy of the model was 0.626681834229004 for players only.

The loss of the model was 0.5711560650259742 for teams only.
The accuracy of the model was 0.7072252294657566 for teams only.

This model was a lot better than the baseline model as for both the player only and team only datasets the loss was lower and the accuracy was higher.

