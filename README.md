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

We then fitlered the rows where the datacompletness column is equal to complete as the rows that aren't complete didn't were missing the values for our columns of interests.


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
  src="assets/kills_bar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot shows the proportion of teams in each league that won a game condtionally on whether they had more kills, the same number of kills, or fewer kills 10 minutes into the game for some of the more popular leagues/competitions. This plot demonstrates the importance of having the greater or an equal amount of kills to the other team as it unlikely to win with fewer kills even at only 10 minutes into the game.

<iframe
  src="assets/one_var_hist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

We know kills are very important on the outcome of a gaame from the previous plot, so we graphed the distribution of the number of kills at 10 minutes for eahc player. We see that almost all the values are 0 and 1, which tells us that kills are rare to come by, which makes them so important as in the game, it is probably the easiest way to gain an advantage.

<iframe
  src="assets/two_var_hist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

We then created a scatter plot of number of kills against the gold differential at 10 minutes with mean highlighted and connected for each number of kills for players. We did this as we established kills are important, but now wanted to see how kills give players an advantage in terms of gold. When you get a kill 300 gold is gained. The general trend of the graph is that mean of points move upward as the number of kills increases. This tells us that the more kills one has the more likely they are to have a larger gold differential. It also tells us that a kill is more important in terms of gold than just the gold gained from the kill, but as well as gold gained from being able to gain more cs by being alive as the difference between the means for many of the nummber of kills is greater than 300. 

<iframe
  src="assets/champ_win_bar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This plot shows the top 5 most popular champions win percentage. We chose to show this as it interesting to see which champions win more often.

```py
print(pivot_table.reset_index().head().to_markdown(index=False))
```

| position   |    Aatrox |       Ahri |     Akali |   Akshan |   Alistar |   Amumu |   Anivia |   Annie |   Aphelios |       Ashe |   Aurelion Sol |      Azir |       Bard |   Bel'Veth |   Blitzcrank |   Brand |    Braum |   Caitlyn |   Camille |   Cassiopeia |   Cho'Gath |     Corki |   Darius |    Diana |   Dr. Mundo |   Draven |     Ekko |   Elise |   Evelynn |    Ezreal |   Fiddlesticks |    Fiora |     Fizz |     Galio |   Gangplank |    Garen |     Gnar |     Gragas |    Graves |       Gwen |   Hecarim |   Heimerdinger |   Illaoi |   Irelia |     Ivern |      Janna |   Jarvan IV |       Jax |    Jayce |      Jhin |     Jinx |   K'Sante |    Kai'Sa |   Kalista |     Karma |   Karthus |   Kassadin |   Katarina |      Kayle |     Kayn |    Kennen |   Kha'Zix |   Kindred |    Kled |   Kog'Maw |   LeBlanc |   Lee Sin |      Leona |   Lillia |   Lissandra |   Lucian |       Lulu |       Lux |   Malphite |   Malzahar |   Maokai |   Master Yi |   Miss Fortune |   Mordekaiser |   Morgana |       Nami |      Nasus |   Nautilus |    Neeko |   Nidalee |     Nilah |   Nocturne |   Nunu & Willump |     Olaf |   Orianna |      Ornn |   Pantheon |     Poppy |     Pyke |   Qiyana |   Quinn |      Rakan |   Rammus |   Rek'Sai |     Rell |   Renata Glasc |   Renekton |    Rengar |   Riven |   Rumble |     Ryze |     Samira |   Sejuani |    Senna |   Seraphine |      Sett |      Shaco |      Shen |    Shyvana |   Singed |     Sion |     Sivir |   Skarner |       Sona |    Soraka |     Swain |      Sylas |   Syndra |   Tahm Kench |   Taliyah |   Talon |      Taric |    Teemo |    Thresh |   Tristana |   Trundle |   Tryndamere |   Twisted Fate |   Twitch |     Udyr |    Urgot |     Varus |    Vayne |    Veigar |   Vel'Koz |       Vex |        Vi |     Viego |    Viktor |   Vladimir |   Volibear |   Warwick |    Wukong |     Xayah |   Xerath |   Xin Zhao |    Yasuo |      Yone |   Yorick |    Yuumi |       Zac |       Zed |     Zeri |     Ziggs |    Zilean |      Zoe |   Zyra |
|:-----------|----------:|-----------:|----------:|---------:|----------:|--------:|---------:|--------:|-----------:|-----------:|---------------:|----------:|-----------:|-----------:|-------------:|--------:|---------:|----------:|----------:|-------------:|-----------:|----------:|---------:|---------:|------------:|---------:|---------:|--------:|----------:|----------:|---------------:|---------:|---------:|----------:|------------:|---------:|---------:|-----------:|----------:|-----------:|----------:|---------------:|---------:|---------:|----------:|-----------:|------------:|----------:|---------:|----------:|---------:|----------:|----------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|---------:|----------:|----------:|----------:|--------:|----------:|----------:|----------:|-----------:|---------:|------------:|---------:|-----------:|----------:|-----------:|-----------:|---------:|------------:|---------------:|--------------:|----------:|-----------:|-----------:|-----------:|---------:|----------:|----------:|-----------:|-----------------:|---------:|----------:|----------:|-----------:|----------:|---------:|---------:|--------:|-----------:|---------:|----------:|---------:|---------------:|-----------:|----------:|--------:|---------:|---------:|-----------:|----------:|---------:|------------:|----------:|-----------:|----------:|-----------:|---------:|---------:|----------:|----------:|-----------:|----------:|----------:|-----------:|---------:|-------------:|----------:|--------:|-----------:|---------:|----------:|-----------:|----------:|-------------:|---------------:|---------:|---------:|---------:|----------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|-----------:|----------:|----------:|----------:|---------:|-----------:|---------:|----------:|---------:|---------:|----------:|----------:|---------:|----------:|----------:|---------:|-------:|
| bot        | -709      |  429       | -679      | nan      |  nan      | nan     | nan      |  nan    |    62.4877 | -135.88    |            nan |  270.667  |   nan      |    nan     |   -1113      |     354 |  57      |   241.325 |  nan      |   -611       |   -157.125 | -296      | -695     |  nan     |   -490      |  412.545 | -134     | nan     |   nan     |  -11.467  |        nan     |  141     |  nan     |   49      |      nan    |  nan     | nan      | -1902      | 226       |  nan       |   138     |       -19.9333 |  nan     |  850     |  -801.667 |   nan      |    122      | nan       | nan      |  -88.3527 | -22.9949 |       nan |  -24.424  |   314.243 | -495.667  |  -183.333 |    nan     |    nan     |  nan       |  nan     |  nan      |  nan      |  104.667  | nan     |  -68.625  |   nan     | -415.833  | -1515.5    |  nan     |   nan       |  109.501 |  -751.5    |   90.4    |  -2662     |    nan     |  nan     |      nan    |        4.75728 |      nan      |  320      | -1722      | -1017      |  -797      |  173.5   |   nan     |   60.0781 |   nan      |         nan      | 527      |  nan      | -1310     |   nan      | -690      | nan      |  nan     |   nan   |   nan      |   nan    |   nan     | nan      |    -1370       |   -222     | -488      |   nan   | nan      | 875      |   -3.66346 | -750.667  | -577.366 |    -189.512 | -148.333  |   nan      |  nan      |   nan      |  nan     | -423     |  -80.7679 |  nan      |  -601.5    | -354.231  | -172.483  |  816.667   |  -59.84  |     -308.789 |  -600.625 |   nan   |   nan      |  nan     | nan       |    147.952 |  nan      |     nan      |      -2012     | -86.8564 |  nan     |  nan     |   19.9159 | -138.523 |  -4.54545 |    510    |  nan      |  nan      | 556       |  -92.5    |   -18      |   nan      |       nan |  -29.3333 |  -4.39118 | -44.2222 |   211.5    | -45.2037 |  nan      |  nan     | -790     |  nan      | -308.333  | -25.0483 | -179.759  | -170.667  | nan      |  nan   |
| jng        |  nan      | -315.5     |  nan      | nan      |  nan      | 218     | nan      |  nan    |   nan      |  nan       |            nan |  nan      |   nan      |    262.796 |    -901.25   |     nan | nan      |   852     |  nan      |    nan       |    nan     |  nan      |  112     |  151.538 |    159.25   |  nan     | -334.667 | 154.222 |  -118.148 | -111      |       -106.125 | -330     |  nan     |  nan      |      355    |  nan     | nan      |   -69.7746 | 197.326   |    1.65532 |   135.432 |       nan      |  742     |  447     |  -348.133 | -1165      |    -81.0894 | 188       | nan      | -287      | nan      |       nan |  nan      |   nan     | -301.667  |   308.068 |    nan     |    nan     |  nan       |  122.948 |  nan      |  -30.6706 |   84.6054 | nan     |  nan      |   nan     |  -14.603  |   nan      |  105.486 |   nan       |  nan     |    57      |   82.5    |   -262     |    nan     | -226.519 |       38.25 |     1304       |      -23.4737 |   86.7895 |   nan      |   nan      |   nan      | -147     |   325.316 |  nan      |    44.8746 |         -33.6667 |  89.7381 |  nan      |  -779.667 |    27.6087 |  -16.4333 | nan      |    4.48  |   nan   |   nan      |  -293.25 |   234.025 | nan      |      nan       |    nan     |  183.829  |   637   | 318.806  | nan      |  nan       |  -81.7827 |  nan     |     142     | -342.25   |    99.4286 | -547      |    36.8333 |  304     |  nan     |  nan      |  -94.7273 |   nan      |  nan      |  nan      |   69.4     |  nan     |      nan     |   142.606 |   276.3 | -1323      | -204     | nan       |   -177     | -141.605  |      83      |        nan     | nan      |  120.284 |  nan     |  nan      |  -99     | nan       |  -1329    | -779      | -134.845  |  48.6307  |  nan      |   nan      |   -40.6231 |       nan |  -91.1729 | nan       | nan      |   -61.3534 | nan      |  nan      |  nan     |  nan     |  -94.1186 |   68.4211 | 349      |  nan      |  nan      | nan      |  nan   |
| mid        |  329.208  |    9.80447 |  -39.6323 | -18.0159 |  nan      | nan     |  75.7872 |  -30.25 |   411      |  nan       |           -207 |   72.0499 |   nan      |    nan     |     nan      |      79 | nan      |   299     | -156.333  |      6.20339 |     26.5   |   38.6931 |  725     | -203.1   |    nan      |  -92.5   |   99.3   | nan     |   nan     |  -34.3077 |        nan     |  446     | -633.333 | -184.26   |      221    |  nan     | -99      |  -275.353  |   7.16667 | -268.667   |  -297     |       -27.7778 | -521     |  117.315 |  -260.727 | -1342      |    nan      | nan       |  57.3636 | -439.667  | nan      |       nan |  -97.8824 |   452.5   | -160.419  |   222.545 |   -163.782 |   -220.083 |    6.44444 |  -34.5   | -740      |  nan      |  nan      | 148.75  |   65.1786 |   110.781 | -414.833  |   nan      |  231.4   |    -1.80976 |  109.252 |  -119.6    |   73.3684 |  -1110.8   |   -240.238 |  nan     |      nan    |      nan       |      308      |  120.375  |   nan      |  -319      |   nan      |  232.211 |   nan     |  nan      |  -274      |         nan      | 440      |  -25.8687 |  -155.268 |   141      |  170      | nan      | -140.706 |  -219   |   nan      |   nan    |   nan     | nan      |      nan       |    100.01  |  nan      |  -446   | -41.6316 |  48.8855 |  nan       |  -70.6667 | -135     |    -135.732 |   50.8462 |   nan      |  nan      | -1059.5    | -376.286 | -143.24  | -203      |  nan      | -1586      | -569.708  |  -71.6264 |  -60.873   |  101.493 |      nan     |  -132.818 |  -381.6 |   nan      | -474     | nan       |    131.329 | -582      |      62.3884 |        211.588 | nan      |  576     | -229     |  -43.8947 |  473.8   | -79.6157  |     61.74 |  -39.4055 |   50.8    | 136.333   |  -26.9747 |  -155.13   |   212      |       nan |  731      | 529       | -31.3109 |  -133      |  21.1776 |  -22.1804 |  nan     | -776     | -719      |   17.88   | -25.9744 |  -54.4565 | -238.232  |  92.1548 |  nan   |
| sup        |  nan      |  nan       |  nan      | nan      |  -52.7232 | -18.061 | nan      |  nan    |   nan      |    9.77011 |            nan |  716      |   -95.5882 |    nan     |     -36.6557 |    -328 | -56.9706 |   126     |  465.333  |    nan       |    739.556 |  nan      |  nan     |  nan     |   1299      |  nan     | 1220     | nan     |   nan     | 1262      |        nan     |  nan     |  nan     |  -10.9651 |      nan    |  nan     | nan      |  -102.217  | nan       |  nan       |   nan     |       455.857  |  nan     |  nan     |  -400     |   -55.8137 |     29.25   | 246       | nan      |  nan      | nan      |       nan |  nan      |   nan     |   31.2162 |   nan     |    nan     |    nan     |  nan       |  nan     |  524      |  nan      |  nan      | nan     |  nan      |   nan     | 1060.22   |   -19.6288 |  nan     |   nan       |  489     |   -35.2473 |  198.714  |    nan     |    nan     | -116.37  |      nan    |      669       |      367      |  188.63   |   -59.8039 |   -33.2308 |   -18.9847 |  236.667 |   nan     |  nan      |   nan      |         nan      | nan      |  nan      |   538.077 |   273.8    |   68      |  70.5627 |  nan     |   nan   |   -58.7082 |   nan    |   nan     | -49.8445 |       -3.10013 |    nan     |  nan      |   nan   | 319.2    | nan      |  nan       |  -45.619  |  300.177 |     336.528 |  138.787  | -1669      |   44.9444 |   nan      |  181.188 |  804.348 |  834      |  nan      |   -75.3402 |    7.7337 |  590.611  | -102.857   |   20     |      145.877 |    60.5   |   nan   |   -66.7473 |  nan     |  -2.28297 |    nan     |   10.4    |     nan      |        nan     | 194      |  nan     |  nan     | 1290      | -140     |  74.25    |    nan    | -988      |   27.6667 | nan       |  420      |   926      |   nan      |       nan | 1115.56   | nan       | -13.4    |   nan      | 753.643  | 1307      |  nan     | -123.409 |  -34      |  nan      | nan      |  164.333  | -111.39   | -46.5    |  181.8 |
| top        |   80.5134 |  594.833   |  -50.1492 | 199.283  |  nan      | nan     | nan      |  nan    |   268.5    |  nan       |            nan | -379      | -1347.5    |    317.5   |     nan      |     nan | nan      |   nan     |  -60.8864 |   -189.5     |   -126     |  205.412  |  302.289 |  nan     |    -99.2857 |   33     | -693.333 | 509     |   nan     |  nan      |        nan     |  102.012 |  nan     |  nan      |      117.65 | -375.056 |  29.5949 |  -209.74   |  45.3809  |  -34.0412  |   nan     |       149.667  |  207.062 |  168.285 | -1952.67  | -1786.71   |   -115.167  |  -4.64655 | 138.237  |  nan      | nan      |      -951 | -584      |   160     | -307.768  |   -98.25  |    nan     |    nan     | -146.707   | -590     |   89.6508 |  nan      | 1669      | 143.057 |  nan      |   805.5   |   93.8148 |  -967      |  146.056 |   489       |  251.545 | -1076.25   | -486      |   -339.165 |    nan     | -263.224 |      nan    |      nan       |       70.9128 |  nan      |  -734      |  -459.211  |   -59      |  568.571 |  -284     | 2148      |   214      |         nan      | 179.51   |  284.5    |  -173.129 |   -56.75   |  -46.0108 | nan      |  724     |  -220.5 | -2398      |  -363    |   nan     | nan      |      nan       |    279.615 |   88.1667 |    85.5 | 293.446  |  44.6154 | -193       | -134.923  |  nan     |    -651.333 |  107.074  | -1914      | -306.821  |  -213.007  | -248.333 | -204.596 |  nan      |  nan      |   nan      | -273.167  |   49.375  |    7.36667 |  nan     |     -215.444 |   nan     |   nan   | -1580.5    |  640.727 | nan       |    961.3   |   82.0952 |     182.191  |        nan     | nan      | -389.462 |  -48.156 | -763      |  157.412 | -12       |    nan    |  263      |   18      |   7.64706 | -141.909  |    16.6964 |   -34.7818 |       924 | -137.748  | nan       | nan      |    59.5    | 163.696  |   56.9346 |  146.231 |  nan     | -157.889  |  nan      | 174.15   |  358      |   78.1765 | nan      |  nan   |


This table shows the average gold diffferential at 10 minutes for each championn in each role. The nan's mean the champion was never played in a given role. We chose to show this as it illustrates which champions are stronger early and give their team an advantange at this point in the game.


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

<iframe
  src="assets/confusion_matrix_player.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The loss of the model was 0.5711560650259742 for teams only.
The accuracy of the model was 0.7072252294657566 for teams only.

This model was a lot better than the baseline model as for both the player only and team only datasets the loss was lower and the accuracy was higher.

<iframe
  src="assets/confusion_matrix_team.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>