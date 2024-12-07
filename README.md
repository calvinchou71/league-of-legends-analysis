# league-of-legends-analysis

# Prediction of the outcomes of League of Legends Matches after 10 minutes have passed.

by Calvin Chou (calchou@umich.edu)

## Introduction

League of Legends is a video game, where the ultimate goal is to be on the winning team. This Portfolio aims to accurately predict the outcome of League of Legends matches as this is the most important variable in the game. The dataset used is from 2022 League of Legends matches for various leagues. It contains information on a number different variables from the game. There are 150180 rows and 161 columns. Given we want to predict the outcome of match, certain variables are more relevant to our prediction than others. 

Our model will predict the result of a league of legends after 10 minutes of the match have passed. The most important feature columns are goldat10, xpat10, csat10, opp_goldat10, opp_xpat10, opp_csat10, golddiffat10, xpdiffat10, csdiffat10, killsat10, assistsat10, deathsat10, opp_killsat10, opp_assistsat10, and opp_deathat10. 
- goldat10 is how much gold a player has at the 10 minute mark. Gold is used to buy items to make your character stronger. \\
- xpat10 is how much xp a player has at the 10 minute mark. XP is used to level up your character, which allows you to improve your abilities and makes your character stronger. \\
- csat10 is how many minions/monsters (cs) a player has killed at the 10 minute mark. cs is similiar to the amount of gold as getting cs gives you gold. \\
\\
- opp_goldat10, opp_xpat10, opp_csat10 is the amount of gold, xp, and cs the opponent has at the 10 minute mark.
\\
*Your opponent is the person on the other team in the same position as you. There are 5 roles in the game, top, jungle, mid, adc, and support.*
\\
- golddiffat10, xpdiffat10, csdiffat10 is the difference between you and your opponent.
\\
- killsat10 is the number of kills you have at 10 minutes. \\
- assistsat10 is the number of assists you have at 10 minutes. Assists are when you don't kill a player on the opposing team, but help in the kill. \\
- deathsat10 is the number of deaths you have at 10 minutes. \\
\\
- opp_killsat10, opp_assistsat10, and opp_deathat10 are the number of kills, assists, and deaths for your opponent.
\\
All of these columns reveal information about the state of the game at the 10 minute mark in the game. 

---

## Cleaning annd EDA

### Cleaning

We seperated the dataset into a datasets. One where the dataset is filtered to contain rows for the entire team rather than a single player. This is useful when we predict the outcomes for an entire team. We then created a dataset for only the players to predict the outcomes for a single player. \\
\\
We then fitlered the rows where the datacompletness column is equal to complete as the rows that aren't complete didn't were missing the values for our columns of interests. \\
\\

Team Dataframe

```py
print(df_clean_team.head().to_markdown(index=False))
```


Player Data Frame

```py
print(df_clean_player.head().to_markdown(index=False))
```

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
print(pivot_table.head().to_markdown(index=False))
```