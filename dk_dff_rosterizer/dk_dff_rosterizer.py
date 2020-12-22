import os
import pandas as pd
import numpy as np
import math
import requests
import gurobipy as gp
from gurobipy import GRB
from sys import argv
from requests import Session
from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()
m=gp.Model(env=env)
    
defColumnSettings = {
    'axis': 1,
    'inplace': True
}

with Session() as s:
    login_data = {"username":"JGearheart21","password":"EECS6893", "remember":"1","token":"0"}
    s.post("https://stathead.com/users/login.cgi",login_data)
    

def encode(df,window):
    # read in file and obtain features and labels
    rowsind = -1*window
    game5stat = rowsind-1
    df = df.dropna()
    df = df.reset_index()
    x = df.iloc[:, rowsind:] # choose last 5 rows (past 4 game history and position)
    y = df.iloc[:, game5stat]  # game 5 stat
    # create one hot encoding for categorical variable
    oh = pd.get_dummies(x.iloc[:, -1])
    # create new features with onehot encoding features
    x = x.drop(x.columns[-1], axis=1)
    x = x.join(oh)

    return x ,y

def optimize(data_df, positions = ['QB', 'RB', 'WR', 'TE', 'DST','FLX']):

    
    position = 'Pos'
    points = 'Fan Points'
    salary = 'Cost'
    playerr = 'Player'   

    player_pt_dict = {}
    player_sal_dict = {}
    
    #player names as index
    for p in data_df.index:
        player_pt_dict[(data_df.loc[p][playerr], data_df.loc[p][position])] = data_df.loc[p][points]
        player_sal_dict[(data_df.loc[p][playerr], data_df.loc[p][position])] = data_df.loc[p][salary]

    combinations, scores = gp.multidict(player_pt_dict)
    combinations2, salary = gp.multidict(player_sal_dict)
    m = gp.Model('OptLineup')
    x = m.addVars(combinations, vtype = GRB.BINARY, name = data_df.Player.tolist())
    m.update()
    
    totplayers = ( 1*('QB' in positions) + 2*('RB' in positions) + 3*('WR' in positions) + 
                  1*('RB' in positions or 'WR' in positions or 'TE' in position) + 1*('TE' in positions) +
                 1*('DST' in positions) )
    
    m.addConstr( ( x.sum('*','*') == totplayers), name='totplayers')
    m.addConstr( ( x.sum('*', 'QB') == 1) , name='num_qb')
    m.addConstr( ( x.sum('*', 'RB') >= 2), name='min_rb')
    m.addConstr( ( x.sum('*', 'WR') >= 3), name='min_wr')
    m.addConstr( ( x.sum('*', 'TE') >= 1), name='min_te')
    m.addConstr( ( x.sum('*', 'DST') == 1), name='num_def')
    m.addConstr( ( x.prod(salary) <= 50000), name='max_sal')
    m.update()
    
    m.setObjective(x.prod(scores), GRB.MAXIMIZE)
    m.write('OptLineup.rlp')
    m.optimize()

    data_df_present = data_df.set_index('Player')
    sal = 50000
    team = []
    for v in m.getVars():
        if v.x == 1:
            team += [v.varName]
    print(team)
    for p in team:
        print('{name} {pos} {pts:g}'.format(name=p, pos = data_df_present.loc[p][position], pts=data_df_present.loc[p][points] ))
    print('Objective Value: {val:g}'.format(val = m.objVal))

    playerInput = input("Remove any of these players? ")
    if playerInput in team:
        data_df=data_df[data_df['Player'] != playerInput]
        optimize(data_df, positions = ['QB', 'RB', 'WR', 'TE', 'FLX', 'DST'])

def rrmodel(x, y):
    # define model
    model = Ridge(alpha=1.0)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    print(cv)
    # evaluate model
    scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # fit model
    model.fit(x, y)

    return model, scores

def get_recent_player_history(df,window):
    checkind = window - 1 
    df = df.dropna().sort_values(['Player', 'Week'], ascending= [True, False])
    players = df.Player.unique()

    game_history = {}
    stats = [20, 19, 21, 24, 23, 'Rec', 26, 27, 9]
    for player in players:
        game_history[player] = {}
        player_history = df.loc[df['Player'] == player]
        if player_history.shape[0] > checkind: # make sure player has enough game history

            for stat in stats:
                # get past 6 games stats and player position vector
                game_history[player][stat] = [player_history.iloc[1][stat],
                                                player_history.iloc[2][stat],
                                                player_history.iloc[3][stat],
                                                player_history.iloc[4][stat]]
                ##JRG
                positions = {'QB':0, 'RB':0, 'TE':0, 'WR':0}
                positions[player_history.iloc[1]['Pos']] = 1
                game_history[player][stat].extend(list(positions.values()))

                # get current position
                game_history[player]['Pos'] = player_history.iloc[0]['Pos']
    
    return game_history

def predict_stats(models, past_stats):
    '''
    :param models: dictionary of models
    :param past_stats: dictionary of a player's of previous game stats
    :return: predicted stat values
    '''
    c=[float(i) for i in past_stats[20]]
    PassTD = models['PassTD'].predict([c])
    c=[float(i) for i in past_stats[19]]
    PassYds = models['PassYds'].predict([c])
    c=[float(i) for i in past_stats[21]]
    INT = models['INT'].predict([c])
    c=[float(i) for i in past_stats[24]]
    RushTD = models['RushTD'].predict([c])
    c=[float(i) for i in past_stats[23]]
    RushYd = models['RushYds'].predict([c])
    c=[float(i) for i in past_stats['Rec']]
    REC = models['Rec'].predict([c])
    c=[float(i) for i in past_stats[26]]
    RecYds = models['RecYds'].predict([c])
    c=[float(i) for i in past_stats[27]]
    RecTD = models['RecTD'].predict([c])
    c=[float(i) for i in past_stats[9]]
    PCD = models['PCD'].predict([c])

    return PassTD, PassYds, INT, RushTD, RushYd, REC, RecYds, RecTD, PCD


def Fantasy_Points(PassTD, PassYd, INT, RushTD, RushYd, REC, RecYd, RecTD, PCD, fumble = 0, fumble_recovered = 0):
    FP = PassTD * 4 + PassYd * 0.04 - 1 * INT + (PassYd>=300) * 3 + \
         RushTD * 6 + RushYd * 0.1 + (RushYd>=100) * 3 + 1 * REC + 0.1 * RecYd + \
         6 * RecTD + (RecYd>=100) * 3 - 1 * math.floor(fumble) + \
         2 * (PCD) + 6 * (math.floor(fumble_recovered))

    return FP[0]

def get_recent_defense_history(df,window):
    checkind = window - 1
    df = df.dropna().sort_values(['Tm', 'Week'], ascending= [True, False])
    players = df.Tm.unique()


    # coeffecients are in the following order: [Game4, Game3, Game2, Game1, CB DB FB FB/DL FB/TE HB LS QB RB RB/WR TE WR WR/RS]

    game_history = {}
    stats = ['TO', 'FR', 'IR', 'KR', 'OR', 'PA', 'PR', 'Sk','Sfty']
    for player in players:
        game_history[player] = {}
        player_history = df.loc[df['Tm'] == player]
        if player_history.shape[0] > checkind: # make sure player has enough game history
            for stat in stats:
                # get past 4 games stats and player position vector
                game_history[player][stat] = [player_history.iloc[1][stat],
                                                  player_history.iloc[2][stat],
                                                  player_history.iloc[3][stat],
                                                  player_history.iloc[4][stat]]

                ##change this JRG
                positions = {'DST':0}
                positions[player_history.iloc[1]['Pos']] = 1
                game_history[player][stat].extend(list(positions.values()))

                # get current game's Draft Kings Fantasy Points earned and position
                game_history[player]['Pos'] = player_history.iloc[0]['Pos']

    return game_history

def predict_defense_stats(models, past_stats):
    '''
    :param models: dictionary of models
    :param past_stats: dictionary of a player's of previous game stats
    :return: predicted stat values
    '''
    c=[float(i) for i in past_stats['TO']]
    TO = models['TO'].predict([c])
    c=[float(i) for i in past_stats['FR']]
    FRTD = models['FR'].predict([c])
    c=[float(i) for i in past_stats['IR']]
    IRTD = models['IR'].predict([c])
    c=[float(i) for i in past_stats['KR']]
    KRTD = models['KR'].predict([c])
    c=[float(i) for i in past_stats['OR']]
    OR = models['OR'].predict([c])
    c=[float(i) for i in past_stats['PA']]
    PA = models['PA'].predict([c])
    c=[float(i) for i in past_stats['PR']]
    PRTD = models['PR'].predict([c])
    c=[float(i) for i in past_stats['Sk']]
    Sk = models['Sk'].predict([c])
    c=[float(i) for i in past_stats['Sfty']]
    Sfty = models['Sfty'].predict([c])

    return TO, FRTD, IRTD, KRTD, OR, PA, PRTD, Sk, Sfty


def defense_Fantasy_Points(TO, FRTD, IRTD, KRTD, OR, PA, PRTD, Sk, Sfty):
    PApts = 0
    if ( PA  == 0 ):
        PApts = 10
    elif ( PA < 7 ):
        PApts = 7
    elif ( PA < 14 ):
        PApts = 4
    elif ( PA < 21 ):
        PApts = 1
    elif ( PA < 28 ):
        PApts = 0
    elif ( PA < 35 ) :
        PApts = -1
    else:
        PApts = -4;

    FP = Sk + 2*Sfty + 2*TO + 6*KRTD + 6*IRTD + 6*OR + 6*PRTD + 6*FRTD + PApts

    return FP[0]


def expected_defense_FP(df, player_data,window):
    """
    :param path: csv file with weekly player game history
    :param player_data: rolling 4 game player game stats
    :return: expected Fantasy Points
    """
    shapeind = window - 1
    df = df.dropna().sort_values(['Tm', 'Week'], ascending= [True, False])
    players = df.Tm.unique()

    defense_FP = {}
    for player in players:
        player_history = df.loc[df['Tm'] == player]
        defense_FP[player] = {}
        if player_history.shape[0] > shapeind:
            TO, FRTD, IRTD, KRTD, OR, PA, PRTD, Sk, Sfty = predict_defense_stats(models_defense, player_data[player])
            FP = defense_Fantasy_Points(TO, FRTD, IRTD, KRTD, OR, PA, PRTD, Sk, Sfty)
            defense_FP[player]['Expected'] = FP
            defense_FP[player]['Pos'] = player_data[player]['Pos']

    return defense_FP

def expected_FP(df, player_data,window):
    """
    :param path: csv file with weekly player game history
    :param player_data: rolling 3 game player game stats
    :return: expected Fantasy Points
    """
    shapeind = window - 1
    df = df.dropna().sort_values(['Player', 'Week'], ascending= [True, False])
    players = df.Player.unique()

    player_FP = {}
    for player in players:
        player_history = df.loc[df['Player'] == player]
        player_FP[player] = {}
        
        if player_history.shape[0] > shapeind:                    
            
            PassTD, PassYds, INT, RushTD, RushYd, Rec, RecYds, RecTD, PCD = predict_stats(models, player_data[player])
            #if (player_history['Pos'].contains('QB')):
            #    print(PassTD)

            FP = Fantasy_Points(PassTD, PassYds, INT, RushTD, RushYd, Rec, RecYds, RecTD, PCD)
            player_FP[player]['Expected'] = FP
            player_FP[player]['Pos'] = player_data[player]['Pos']

    return player_FP

def reprocess_players(df,window):
    df = df.dropna()
    df = df[df['Tm'] != 'Pos']
    exported = df[['Player','Tm','Pos']]
    gf = exported.drop_duplicates(inplace=False)
    gf = gf.reset_index(drop=True)

    #JRG
    OverallRecTD = pd.DataFrame(columns=["Player","Tm","RecTDN","RecTDN_1","RecTDN_2","RecTDN_3","RecTDN_4","Pos"])
    OverallRec = pd.DataFrame(columns=["Player","Tm","RecN","RecN_1","RecN_2","RecN_3","RecN_4","Pos"])
    OverallRecYds = pd.DataFrame(columns=["Player","Tm","RecYdsN","RecYdsN_1","RecYdsN_2","RecYdsN_3","RecYdsN_4","Pos"])
    OverallPassTD = pd.DataFrame(columns=["Player","Tm","PassTDN","PassTDN_1","PassTDN_2","PassTDN_3","PassTDN_4","Pos"])
    OverallPassYds = pd.DataFrame(columns=["Player","Tm","PassYdsN","PassYdsN_1","PassYdsN_2","PassYdsN_3","PassYdsN_4","Pos"])
    OverallRushTD = pd.DataFrame(columns=["Player","Tm","RushTDN","RushTDN_1","RushTDN_2","RushTDN_3","RushTDN_4","Pos"])
    OverallRushYds = pd.DataFrame(columns=["Player","Tm","RushYdsN","RushYdsN_1","RushYdsN_2","RushYdsN_3","RushYdsN_4","Pos"])
    OverallPCD = pd.DataFrame(columns=["Player","Tm","PCDN","PCDN_1","PCDN_2","PCDN_3","PCDN_4","Pos"])
    OverallFR = pd.DataFrame(columns=["Player","Tm","FRN","FRN_1","FRN_2","FRN_3","FRN_4","Pos"])
    OverallFL = pd.DataFrame(columns=["Player","Tm","FLN","FLN_1","FLN_2","FLN_3","FLN_4","Pos"])
    OverallINT = pd.DataFrame(columns=["Player","Tm","INTN","INTN_1","INTN_2","INTN_3","INTN_4","Pos"])
    

#JRG
    for gf_index in range(0,len(gf.index)):
        PlayerTableINT = pd.DataFrame(columns=["Player","Tm","INTN","INTN_1","INTN_2","INTN_3","INTN_4","Pos"])
        PlayerTableFL = pd.DataFrame(columns=["Player","Tm","FLN","FLN_1","FLN_2","FLN_3","FLN_4","Pos"])
        PlayerTablePCD = pd.DataFrame(columns=["Player","Tm","PCDN","PCDN_1","PCDN_2","PCDN_3","PCDN_4","Pos"])
        PlayerTableFR = pd.DataFrame(columns=["Player","Tm","FRN","FRN_1","FRN_2","FRN_3","FRN_4","Pos"])
        PlayerTableRushYds = pd.DataFrame(columns=["Player","Tm","RushYdsN","RushYdsN_1","RushYdsN_2","RushYdsN_3","RushYdsN_4","Pos"])
        PlayerTableRushTD = pd.DataFrame(columns=["Player","Tm","RushTDN","RushTDN_1","RushTDN_2","RushTDN_3","RushTDN_4","Pos"])
        PlayerTablePassTD = pd.DataFrame(columns=["Player","Tm","PassTDN","PassTDN_1","PassTDN_2","PassTDN_3","PassTDN_4","Pos"])
        PlayerTablePassYds = pd.DataFrame(columns=["Player","Tm","PassYdsN","PassYdsN_1","PassYdsN_2","PassYdsN_3","PassYdsN_4","Pos"])
        PlayerTableRecYds = pd.DataFrame(columns=["Player","Tm","RecYdsN","RecYdsN_1","RecYdsN_2","RecYdsN_3","RecYdsN_4","Pos"])
        PlayerTableRec = pd.DataFrame(columns=["Player","Tm","RecN","RecN_1","RecN_2","RecN_3","RecN_4","Pos"])
        PlayerTableRecTD = pd.DataFrame(columns=["Player","Tm","RecTDN","RecTDN_1","RecTDN_2","RecTDN_3","RecTDN_4","Pos"])

        idx = np.where((df['Player']==gf.loc[gf_index]['Player']) & (df['Tm']==gf.loc[gf_index]['Tm']) & (df['Pos']==gf.loc[gf_index]['Pos']))
        
        player_only = df.loc[idx]
        player_games = len(player_only.index)
        player_only = player_only.reset_index(drop=True)
        
       
        
        if ((player_games >= window)&((player_only.loc[0,'Pos'] == 'RB') or (player_only.loc[0,'Pos'] == 'WR')or (player_only.loc[0,'Pos'] == 'QB')or (player_only.loc[0,'Pos'] == 'TE'))):
            player_only = player_only.reset_index(drop=True)
            player_only = player_only.sort_values(by=['Week'],ascending=False)
            

            NrowsPlayer = player_games-window+1
            for stat_ind in range(0,NrowsPlayer-1):

                #JRG

                PlayerTableRecTD.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableRecTD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableRecTD.loc[stat_ind,'RecTDN']=player_only.loc[stat_ind][27]
                PlayerTableRecTD.loc[stat_ind,'RecTDN_1']=player_only.loc[stat_ind+1][27]
                PlayerTableRecTD.loc[stat_ind,'RecTDN_2']=player_only.loc[stat_ind+2][27]
                PlayerTableRecTD.loc[stat_ind,'RecTDN_3']=player_only.loc[stat_ind+3][27]
                PlayerTableRecTD.loc[stat_ind,'RecTDN_4']=player_only.loc[stat_ind+4][27]
                PlayerTableRecTD.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']
                

                PlayerTableRec.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableRec.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableRec.loc[stat_ind,'RecN']=player_only.loc[stat_ind,'Rec']
                PlayerTableRec.loc[stat_ind,'RecN_1']=player_only.loc[stat_ind+1,'Rec']
                PlayerTableRec.loc[stat_ind,'RecN_2']=player_only.loc[stat_ind+2,'Rec']
                PlayerTableRec.loc[stat_ind,'RecN_3']=player_only.loc[stat_ind+3,'Rec']
                PlayerTableRec.loc[stat_ind,'RecN_4']=player_only.loc[stat_ind+4,'Rec']
                PlayerTableRec.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTableRecYds.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableRecYds.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableRecYds.loc[stat_ind,'RecYdsN']=player_only.loc[stat_ind][27]
                PlayerTableRecYds.loc[stat_ind,'RecYdsN_1']=player_only.loc[stat_ind+1][26]
                PlayerTableRecYds.loc[stat_ind,'RecYdsN_2']=player_only.loc[stat_ind+2][26]
                PlayerTableRecYds.loc[stat_ind,'RecYdsN_3']=player_only.loc[stat_ind+3][26]
                PlayerTableRecYds.loc[stat_ind,'RecYdsN_4']=player_only.loc[stat_ind+4][26]
                PlayerTableRecYds.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']
     
                PlayerTableRushTD.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableRushTD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableRushTD.loc[stat_ind,'RushTDN']=player_only.loc[stat_ind][24]
                PlayerTableRushTD.loc[stat_ind,'RushTDN_1']=player_only.loc[stat_ind+1][24]
                PlayerTableRushTD.loc[stat_ind,'RushTDN_2']=player_only.loc[stat_ind+2][24]
                PlayerTableRushTD.loc[stat_ind,'RushTDN_3']=player_only.loc[stat_ind+3][24]
                PlayerTableRushTD.loc[stat_ind,'RushTDN_4']=player_only.loc[stat_ind+4][24]
                PlayerTableRushTD.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTableRushYds.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableRushYds.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableRushYds.loc[stat_ind,'RushYdsN']=player_only.loc[stat_ind][23]
                PlayerTableRushYds.loc[stat_ind,'RushYdsN_1']=player_only.loc[stat_ind+1][23]
                PlayerTableRushYds.loc[stat_ind,'RushYdsN_2']=player_only.loc[stat_ind+2][23]
                PlayerTableRushYds.loc[stat_ind,'RushYdsN_3']=player_only.loc[stat_ind+3][23]
                PlayerTableRushYds.loc[stat_ind,'RushYdsN_4']=player_only.loc[stat_ind+4][23]
                PlayerTableRushYds.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTablePassYds.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTablePassYds.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTablePassYds.loc[stat_ind,'PassYdsN']=player_only.loc[stat_ind][19]
                PlayerTablePassYds.loc[stat_ind,'PassYdsN_1']=player_only.loc[stat_ind+1][19]
                PlayerTablePassYds.loc[stat_ind,'PassYdsN_2']=player_only.loc[stat_ind+2][19]
                PlayerTablePassYds.loc[stat_ind,'PassYdsN_3']=player_only.loc[stat_ind+3][19]
                PlayerTablePassYds.loc[stat_ind,'PassYdsN_4']=player_only.loc[stat_ind+4][19]
                PlayerTablePassYds.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTablePassTD.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTablePassTD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTablePassTD.loc[stat_ind,'PassTDN']=player_only.loc[stat_ind][20]
                PlayerTablePassTD.loc[stat_ind,'PassTDN_1']=player_only.loc[stat_ind+1][20]
                PlayerTablePassTD.loc[stat_ind,'PassTDN_2']=player_only.loc[stat_ind+2][20]
                PlayerTablePassTD.loc[stat_ind,'PassTDN_3']=player_only.loc[stat_ind+3][20]
                PlayerTablePassTD.loc[stat_ind,'PassTDN_4']=player_only.loc[stat_ind+4][20]
                PlayerTablePassTD.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTableFR.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableFR.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableFR.loc[stat_ind,'FRN']=player_only.loc[stat_ind,'FR']
                PlayerTableFR.loc[stat_ind,'FRN_1']=player_only.loc[stat_ind+1,'FR']
                PlayerTableFR.loc[stat_ind,'FRN_2']=player_only.loc[stat_ind+2,'FR']
                PlayerTableFR.loc[stat_ind,'FRN_3']=player_only.loc[stat_ind+3,'FR']
                PlayerTableFR.loc[stat_ind,'FRN_4']=player_only.loc[stat_ind+4,'FR']
                PlayerTableFR.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTableFL.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableFL.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableFL.loc[stat_ind,'FLN']=player_only.loc[stat_ind,'FL']
                PlayerTableFL.loc[stat_ind,'FLN_1']=player_only.loc[stat_ind+1,'FL']
                PlayerTableFL.loc[stat_ind,'FLN_2']=player_only.loc[stat_ind+2,'FL']
                PlayerTableFL.loc[stat_ind,'FLN_3']=player_only.loc[stat_ind+3,'FL']
                PlayerTableFL.loc[stat_ind,'FLN_4']=player_only.loc[stat_ind+4,'FL']
                PlayerTableFL.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTablePCD.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTablePCD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTablePCD.loc[stat_ind,'PCDN']=player_only.loc[stat_ind][9]
                PlayerTablePCD.loc[stat_ind,'PCDN_1']=player_only.loc[stat_ind+1][9]
                PlayerTablePCD.loc[stat_ind,'PCDN_2']=player_only.loc[stat_ind+2][9]
                PlayerTablePCD.loc[stat_ind,'PCDN_3']=player_only.loc[stat_ind+3][9]
                PlayerTablePCD.loc[stat_ind,'PCDN_4']=player_only.loc[stat_ind+4][9]
                PlayerTablePCD.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']

                PlayerTableINT.loc[stat_ind,'Player']=player_only.loc[stat_ind,'Player']
                PlayerTableINT.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableINT.loc[stat_ind,'INTN']=player_only.loc[stat_ind][21]
                PlayerTableINT.loc[stat_ind,'INTN_1']=player_only.loc[stat_ind+1][21]
                PlayerTableINT.loc[stat_ind,'INTN_2']=player_only.loc[stat_ind+2][21]
                PlayerTableINT.loc[stat_ind,'INTN_3']=player_only.loc[stat_ind+3][21]
                PlayerTableINT.loc[stat_ind,'INTN_4']=player_only.loc[stat_ind+4][21]
                PlayerTableINT.loc[stat_ind,'Pos']=player_only.loc[stat_ind,'Pos']
                
            OverallRecTD = pd.concat([OverallRecTD,PlayerTableRecTD])
            OverallRec = pd.concat([OverallRec,PlayerTableRec])
            OverallRecYds = pd.concat([OverallRecYds,PlayerTableRecYds])
            OverallRushTD = pd.concat([OverallRushTD,PlayerTableRushTD])
            OverallRushYds = pd.concat([OverallRushYds,PlayerTableRushYds])
            OverallPassTD = pd.concat([OverallPassTD,PlayerTablePassTD])
            OverallPassYds = pd.concat([OverallPassYds,PlayerTablePassYds])
            OverallFR = pd.concat([OverallFR,PlayerTableFR])
            OverallFL = pd.concat([OverallFL,PlayerTableFL])
            OverallPCD = pd.concat([OverallPCD,PlayerTablePCD])
            OverallINT = pd.concat([OverallINT,PlayerTableINT])

    return OverallRecTD, OverallRec, OverallRecYds, OverallRushTD, OverallRushYds, OverallPassTD, OverallPassYds, OverallFR, OverallFL, OverallPCD, OverallINT

def reprocess_defenses(df,window):
    
    exported = df['Tm']
    gf = exported.drop_duplicates(inplace=False)
    gf = gf.reset_index(drop=True)

    #JRG

    OverallPA = pd.DataFrame(columns=["Tm","Tm2","PAN","PAN_1","PAN_2","PAN_3","PAN_4","Pos"])
    OverallSfty = pd.DataFrame(columns=["Tm","Tm2","SftyN","SftyN_1","SftyN_2","SftyN_3","SftyN_4","Pos"])
    OverallTO = pd.DataFrame(columns=["Tm","Tm2","TON","TON_1","TON_2","TON_3","TON_4","Pos"])
    OverallSk = pd.DataFrame(columns=["Tm","Tm2","SkN","SkN_1","SkN_2","SkN_3","SkN_4","Pos"])
    OverallKRTD = pd.DataFrame(columns=["Tm","Tm2","KRTDN","KRTDN_1","KRTDN_2","KRTDN_3","KRTDN_4","Pos"])
    OverallPRTD = pd.DataFrame(columns=["Tm","Tm2","PRTDN","PRTDN_1","PRTDN_2","PRTDN_3","PRTDN_4","Pos"])
    OverallIRTD = pd.DataFrame(columns=["Tm","Tm2","IRTDN","IRTDN_1","IRTDN_2","IRTDN_3","IRTDN_4","Pos"])
    OverallFRTD = pd.DataFrame(columns=["Tm","Tm2","FRTDN","FRTDN_1","FRTDN_2","FRTDN_3","FRTDN_4","Pos"])
    OverallOR = pd.DataFrame(columns=["Tm","Tm2","ORN","ORN_1","ORN_2","ORN_3","ORN_4","Pos"])
    
    for gf_index in range(0,len(gf.index)):
        PlayerTablePA = pd.DataFrame(columns=["Tm","Tm2","PAN","PAN_1","PAN_2","PAN_3","PAN_4","Pos"])
        PlayerTableSfty = pd.DataFrame(columns=["Tm","Tm2","SftyN","SftyN_1","SftyN_2","SftyN_3","SftyN_4","Pos"])
        PlayerTableTO = pd.DataFrame(columns=["Tm","Tm2","TON","TON_1","TON_2","TON_3","TON_4","Pos"])
        PlayerTableSk = pd.DataFrame(columns=["Tm","Tm2","SkN","SkN_1","SkN_2","SkN_3","SkN_4","Pos"])
        PlayerTableKRTD = pd.DataFrame(columns=["Tm","Tm2","KRTDN","KRTDN_1","KRTDN_2","KRTDN_3","KRTDN_4","Pos"])
        PlayerTablePRTD = pd.DataFrame(columns=["Tm","Tm2","PRTDN","PRTDN_1","PRTDN_2","PRTDN_3","PRTDN_4","Pos"])
        PlayerTableIRTD = pd.DataFrame(columns=["Tm","Tm2","IRTDN","IRTDN_1","IRTDN_2","IRTDN_3","IRTDN_4","Pos"])
        PlayerTableOR = pd.DataFrame(columns=["Tm","Tm2","ORN","ORN_1","ORN_2","ORN_3","ORN_4","Pos"])
        PlayerTableFRTD = pd.DataFrame(columns=["Tm","Tm2","FRTDN","FRTDN_1","FRTDN_2","FRTDN_3","FRTDN_4","Pos"])
        
        idx = np.where(df['Tm'] == gf.loc[gf_index])
        player_only = df.loc[idx]
        player_only = player_only.sort_values(by=['Week'],ascending=False)
        player_games = len(player_only.index)
        player_only = player_only.reset_index(drop=True)
        
        if (player_games >= window):
            NrowsPlayer = player_games-window+1
            for stat_ind in range(0,NrowsPlayer-1):
                #JRG
                PlayerTablePA.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTablePA.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTablePA.loc[stat_ind,'PAN']=player_only.loc[stat_ind,'PA']
                PlayerTablePA.loc[stat_ind,'PAN_1']=player_only.loc[stat_ind+1,'PA']
                PlayerTablePA.loc[stat_ind,'PAN_2']=player_only.loc[stat_ind+2,'PA']
                PlayerTablePA.loc[stat_ind,'PAN_3']=player_only.loc[stat_ind+3,'PA']
                PlayerTablePA.loc[stat_ind,'PAN_4']=player_only.loc[stat_ind+4,'PA']
                PlayerTablePA.loc[stat_ind,'Pos']='DST'

                PlayerTableSfty.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableSfty.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTableSfty.loc[stat_ind,'SftyN']=player_only.loc[stat_ind,'Sfty']
                PlayerTableSfty.loc[stat_ind,'SftyN_1']=player_only.loc[stat_ind+1,'Sfty']
                PlayerTableSfty.loc[stat_ind,'SftyN_2']=player_only.loc[stat_ind+2,'Sfty']
                PlayerTableSfty.loc[stat_ind,'SftyN_3']=player_only.loc[stat_ind+3,'Sfty']
                PlayerTableSfty.loc[stat_ind,'SftyN_4']=player_only.loc[stat_ind+4,'Sfty']
                PlayerTableSfty.loc[stat_ind,'Pos']='DST'

                PlayerTableTO.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableTO.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTableTO.loc[stat_ind,'TON']=player_only.loc[stat_ind,'TO']
                PlayerTableTO.loc[stat_ind,'TON_1']=player_only.loc[stat_ind+1,'TO']
                PlayerTableTO.loc[stat_ind,'TON_2']=player_only.loc[stat_ind+2,'TO']
                PlayerTableTO.loc[stat_ind,'TON_3']=player_only.loc[stat_ind+3,'TO']
                PlayerTableTO.loc[stat_ind,'TON_4']=player_only.loc[stat_ind+4,'TO']
                PlayerTableTO.loc[stat_ind,'Pos']='DST'
     
                PlayerTableSk.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableSk.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTableSk.loc[stat_ind,'SkN']=player_only.loc[stat_ind,'Sk']
                PlayerTableSk.loc[stat_ind,'SkN_1']=player_only.loc[stat_ind+1,'Sk']
                PlayerTableSk.loc[stat_ind,'SkN_2']=player_only.loc[stat_ind+2,'Sk']
                PlayerTableSk.loc[stat_ind,'SkN_3']=player_only.loc[stat_ind+3,'Sk']
                PlayerTableSk.loc[stat_ind,'SkN_4']=player_only.loc[stat_ind+4,'Sk']
                PlayerTableSk.loc[stat_ind,'Pos']='DST'

                PlayerTableKRTD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableKRTD.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTableKRTD.loc[stat_ind,'KRTDN']=player_only.loc[stat_ind,'KR']
                PlayerTableKRTD.loc[stat_ind,'KRTDN_1']=player_only.loc[stat_ind+1,'KR']
                PlayerTableKRTD.loc[stat_ind,'KRTDN_2']=player_only.loc[stat_ind+2,'KR']
                PlayerTableKRTD.loc[stat_ind,'KRTDN_3']=player_only.loc[stat_ind+3,'KR']
                PlayerTableKRTD.loc[stat_ind,'KRTDN_4']=player_only.loc[stat_ind+4,'KR']
                PlayerTableKRTD.loc[stat_ind,'Pos']='DST'

                PlayerTablePRTD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTablePRTD.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTablePRTD.loc[stat_ind,'PRTDN']=player_only.loc[stat_ind,'PR']
                PlayerTablePRTD.loc[stat_ind,'PRTDN_1']=player_only.loc[stat_ind+1,'PR']
                PlayerTablePRTD.loc[stat_ind,'PRTDN_2']=player_only.loc[stat_ind+2,'PR']
                PlayerTablePRTD.loc[stat_ind,'PRTDN_3']=player_only.loc[stat_ind+3,'PR']
                PlayerTablePRTD.loc[stat_ind,'PRTDN_4']=player_only.loc[stat_ind+4,'PR']
                PlayerTablePRTD.loc[stat_ind,'Pos']='DST'

                PlayerTableIRTD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableIRTD.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTableIRTD.loc[stat_ind,'IRTDN']=player_only.loc[stat_ind,'IR']
                PlayerTableIRTD.loc[stat_ind,'IRTDN_1']=player_only.loc[stat_ind+1,'IR']
                PlayerTableIRTD.loc[stat_ind,'IRTDN_2']=player_only.loc[stat_ind+2,'IR']
                PlayerTableIRTD.loc[stat_ind,'IRTDN_3']=player_only.loc[stat_ind+3,'IR']
                PlayerTableIRTD.loc[stat_ind,'IRTDN_4']=player_only.loc[stat_ind+4,'IR']
                PlayerTableIRTD.loc[stat_ind,'Pos']='DST'

                PlayerTableFRTD.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableFRTD.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTableFRTD.loc[stat_ind,'FRTDN']=player_only.loc[stat_ind,'FR']
                PlayerTableFRTD.loc[stat_ind,'FRTDN_1']=player_only.loc[stat_ind+1,'FR']
                PlayerTableFRTD.loc[stat_ind,'FRTDN_2']=player_only.loc[stat_ind+2,'FR']
                PlayerTableFRTD.loc[stat_ind,'FRTDN_3']=player_only.loc[stat_ind+3,'FR']
                PlayerTableFRTD.loc[stat_ind,'FRTDN_4']=player_only.loc[stat_ind+4,'FR']
                PlayerTableFRTD.loc[stat_ind,'Pos']='DST'

                PlayerTableOR.loc[stat_ind,'Tm']=player_only.loc[stat_ind,'Tm']
                PlayerTableOR.loc[stat_ind,'Tm2']=player_only.loc[stat_ind,'Tm']
                PlayerTableOR.loc[stat_ind,'ORN']=player_only.loc[stat_ind,'OR']
                PlayerTableOR.loc[stat_ind,'ORN_1']=player_only.loc[stat_ind+1,'OR']
                PlayerTableOR.loc[stat_ind,'ORN_2']=player_only.loc[stat_ind+2,'OR']
                PlayerTableOR.loc[stat_ind,'ORN_3']=player_only.loc[stat_ind+3,'OR']
                PlayerTableOR.loc[stat_ind,'ORN_4']=player_only.loc[stat_ind+4,'OR']
                PlayerTableOR.loc[stat_ind,'Pos']='DST'


            OverallPA = pd.concat([OverallPA,PlayerTablePA])
            OverallSfty = pd.concat([OverallSfty,PlayerTableSfty])
            OverallTO = pd.concat([OverallTO,PlayerTableTO])
            OverallSk = pd.concat([OverallSk,PlayerTableSk])
            OverallKRTD = pd.concat([OverallKRTD,PlayerTableKRTD])
            OverallPRTD = pd.concat([OverallPRTD,PlayerTablePRTD])
            OverallIRTD = pd.concat([OverallIRTD,PlayerTableIRTD])
            OverallFRTD = pd.concat([OverallFRTD,PlayerTableFRTD])
            OverallOR = pd.concat([OverallOR,PlayerTableOR])

    return OverallPA, OverallSfty, OverallTO, OverallKRTD, OverallPRTD, OverallIRTD, OverallFRTD, OverallOR, OverallSk

def scrapesalary(weekNum):
    url = "https://www.footballdiehards.com/fantasyfootball/dailygames/Draftkings-Salary-data.cfm"
    login_data = {'qweek':weekNum}
    response = requests.post(url, data=login_data)
    soup = BeautifulSoup(response.content, 'html.parser')    
    table = soup.find('table')
    df = pd.read_html(str(table))[0]
    df.columns = df.columns.droplevel(level = 0)
    df.drop(['Rank','Factor','Score','week','year'], **defColumnSettings)
    defenses=df[df['Pos'] == 'DST']
    df = df[df['Pos'] != 'DST']
    df[['Last','First']]=df.Player.str.split(pat=', ',expand=True)
    df.drop(['Player'], **defColumnSettings)
    df['Player'] = df['First'].str.cat(df['Last'],sep=" ")
    df.drop(['First','Last'], **defColumnSettings)
    first_col=df.pop('Player')
    df.insert(0,'Player',first_col)
    df=pd.concat([df,defenses],ignore_index=True)
    return df, defenses


###__init__##
##scrape salary
weekNum = input('What week of the 2020 NFL season to predict?: 7-15 ')
weekNum = int(weekNum)
player_salary, defense_salary = scrapesalary(weekNum)

##scrape player performances
print("Scraping Historical Player Performances...")
year = 2020
weekmin = 1
weekmax = weekNum-1
window = 5
#JRG

URL1 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=0
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL2 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=100
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL3 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=200
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL4 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=300
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL5 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=400
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL6 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=500
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL7 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=600
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL8 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=700
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL9 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=800
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL10 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=900
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL11 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1000
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL12 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1100
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL13 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1200
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL14 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1300
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL15 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1400
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL16 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1500
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL17 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1600
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL18 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1700
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL19 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1800
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL20 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=1900
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL21 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2000
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL22 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2100
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL23 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2200
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL24 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2300
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL25 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2400
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL26 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2500
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL27 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2600
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL28 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2700
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL29 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2800
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL30 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=2900
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL31 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3000
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL32 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3100
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL33 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3200
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL34 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3300
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL35 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3400
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL36 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3500
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL37 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3600
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL38 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3700
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL39 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3800
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL40 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=3900
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL41 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=4000
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL42 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=4100
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL43 = """
https://stathead.com/football/pgl_finder.cgi?request=1&game_num_max={weekmax}&week_num_max=16&order_by=draftkings_points&season_start=1&qb_gwd=0&order_by_asc=0&qb_comeback=0&week_num_min=1&game_num_min=1&year_min={year}&match=game&year_max={year}&season_end=-1&age_min=0&game_type=R&age_max=99&positions[]=qb&positions[]=rb&positions[]=wr&positions[]=te&cstat[1]=draftkings_points&ccomp[1]=gt&cval[1]=0&cstat[2]=two_pt_md&ccomp[2]=gt&cval[2]=0&cstat[3]=fumbles_rec&ccomp[3]=gt&cval[3]=0&offset=4200
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

i = [URL2,URL3,URL4,URL5,URL6,URL7,URL8,URL9,URL10,URL11,URL12,URL13,URL14,URL15,URL16,URL17,URL18,URL19,URL20,URL21,URL22,URL23,URL24,URL25,URL26,URL27,URL28,URL29,URL30,URL31,URL32,URL33,URL34,URL35,URL36,URL37,URL38,URL39,URL40,URL41,URL42,URL43]
   
###player data
response = s.get(URL1)
soup = BeautifulSoup(response.content, 'html.parser')    
table = soup.find('table', {'id': 'results'})
df = pd.read_html(str(table))[0]
df.columns = df.columns.droplevel(level = 0)
df.drop(['Rk','Lg','Date','XPM','XPA','XP%','FGM','FGA','FG%','Sfty','FantPt','PPR','FDPt','Fmb','FF'], **defColumnSettings)
df = df[df['Pos'] != 'Pos']
qf = df[df['Pos'] == 'QB']
rf = df[df['Pos'] == 'RB']
wf = df[df['Pos'] == 'WR']
tf = df[df['Pos'] == 'TE']
df=[]
df = pd.concat([qf,rf,wf,tf])


df.fillna(0, inplace=True)

for x in i:
    try: 
        response = s.get(x)
        soup = BeautifulSoup(response.content, 'html.parser')    
        table = soup.find('table', {'id': 'results'})
        cf = pd.read_html(str(table))[0]
        cf.columns = cf.columns.droplevel(level = 0)
        cf.drop(['Rk','Lg','Date','XPM','XPA','XP%','FGM','FGA','FG%','Sfty','FantPt','PPR','FDPt','Fmb','FF'], **defColumnSettings)
        cf = cf[cf['Pos'] != 'Pos']
        qf = cf[cf['Pos'] == 'QB']
        rf = cf[cf['Pos'] == 'RB']
        wf = cf[cf['Pos'] == 'WR']
        tf = cf[cf['Pos'] == 'TE']
        cf=[]
        cf = pd.concat([qf,rf,wf,tf])

        cf.fillna(0, inplace=True)
        df=pd.concat([df,cf])
    except:
        pass
    
Tm=df['Tm']
df.drop(labels=['Tm'],axis=1,inplace=True)
df.insert(0,'Tm',Tm)

Week=df['Week']
df.drop(labels=['Week'],axis=1,inplace=True)
df.insert(1,'Week',Week)


##conditions
URLA = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&temperature=10&offset=0
""".format(weekmax=weekmax)

URLB = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&temperature=10&offset=100
""".format(weekmax=weekmax)

URLC = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&temperature=10&offset=200
""".format(weekmax=weekmax)

URLD = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&temperature=10&offset=300
""".format(weekmax=weekmax)

URLE = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&temperature=10&offset=400
""".format(weekmax=weekmax)

j = [URLB,URLC,URLD, URLE]

response = s.get(URLA)
soup = BeautifulSoup(response.content, 'html.parser')    
table = soup.find('table', {'id': 'results'})
gf = pd.read_html(str(table))[0]
gf.columns = gf.columns.droplevel(level = 0)
gf.drop(['Rk','Year','Date','Time','G#','Result','OT','Day','Opp'], **defColumnSettings)
gf = gf[gf['LTime'] != 'LTime']

for x in j:
    try:
        response = s.get(x)
        soup = BeautifulSoup(response.content, 'html.parser')    
        table = soup.find('table', {'id': 'results'})
        hf = pd.read_html(str(table))[0]
        hf.columns = hf.columns.droplevel(level = 0)
        hf.drop(['Rk','Year','Date','Time','G#','Result','OT','Day','Opp'], **defColumnSettings)
        hf = hf[hf['LTime'] != 'LTime']
        gf=pd.concat([gf,hf])
            
    except:
        pass
        
gf.rename({'Unnamed: 6_level_1':'SYMB'}, **defColumnSettings)
gf['HomeAway'] = np.where(gf['SYMB']!= '@','Home','Away')
    
gf =gf.drop(['SYMB'],axis=1)
combinedf = df.merge(gf, how='inner', left_on=['Tm','Week'], right_on=['Tm','Week'])
combinedf = combinedf.drop(['Unnamed: 7_level_1'],axis=1)

Player=combinedf['Player']
combinedf.drop(labels=['Player'],axis=1,inplace=True)
combinedf.insert(0,'Player',Player)
Pos=combinedf['Pos']
combinedf.drop(labels=['Pos'],axis=1,inplace=True)
combinedf.insert(1,'Pos', Pos)

##scrape defense performances
print("Scraping Historical Defense Performances...")
URL1 = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=lt&game_num_max=99&week_num_max=99&order_by=takeaways&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&cstat[1]=points_opp&ccomp[1]=gt&cval[1]=0&cstat[2]=safety_md&ccomp[2]=gt&cval[2]=0&cstat[3]=pass_sacked_opp&ccomp[3]=gt&cval[3]=0&cstat[4]=other_td_tgl&ccomp[4]=gt&cval[4]=0&offset=0
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL2 = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=lt&game_num_max=99&week_num_max=99&order_by=takeaways&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&cstat[1]=points_opp&ccomp[1]=gt&cval[1]=0&cstat[2]=safety_md&ccomp[2]=gt&cval[2]=0&cstat[3]=pass_sacked_opp&ccomp[3]=gt&cval[3]=0&cstat[4]=other_td_tgl&ccomp[4]=gt&cval[4]=0&offset=100
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL3 = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=lt&game_num_max=99&week_num_max=99&order_by=takeaways&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&cstat[1]=points_opp&ccomp[1]=gt&cval[1]=0&cstat[2]=safety_md&ccomp[2]=gt&cval[2]=0&cstat[3]=pass_sacked_opp&ccomp[3]=gt&cval[3]=0&cstat[4]=other_td_tgl&ccomp[4]=gt&cval[4]=0&offset=200
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL4 = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=lt&game_num_max=99&week_num_max=99&order_by=takeaways&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&cstat[1]=points_opp&ccomp[1]=gt&cval[1]=0&cstat[2]=safety_md&ccomp[2]=gt&cval[2]=0&cstat[3]=pass_sacked_opp&ccomp[3]=gt&cval[3]=0&cstat[4]=other_td_tgl&ccomp[4]=gt&cval[4]=0&offset=300
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL5 = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=lt&game_num_max=99&week_num_max=99&order_by=takeaways&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&cstat[1]=points_opp&ccomp[1]=gt&cval[1]=0&cstat[2]=safety_md&ccomp[2]=gt&cval[2]=0&cstat[3]=pass_sacked_opp&ccomp[3]=gt&cval[3]=0&cstat[4]=other_td_tgl&ccomp[4]=gt&cval[4]=0&offset=400
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URL6 = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=lt&game_num_max=99&week_num_max=99&order_by=takeaways&match=game&year_max=2020&order_by_asc=0&week_num_min=1&game_type=R&game_num_min=0&year_min=2020&cstat[1]=points_opp&ccomp[1]=gt&cval[1]=0&cstat[2]=safety_md&ccomp[2]=gt&cval[2]=0&cstat[3]=pass_sacked_opp&ccomp[3]=gt&cval[3]=0&cstat[4]=other_td_tgl&ccomp[4]=gt&cval[4]=0&offset=500
""".format(year=year, weekmin=weekmin, weekmax=weekmax)


defColumnSettings = {
    'axis': 1,
    'inplace': True
}

i = [URL2,URL3,URL4,URL5,URL6]

response = s.get(URL1)
soup = BeautifulSoup(response.content, 'html.parser')    
table = soup.find('table', {'id': 'results'})
df = pd.read_html(str(table))[0]
df.columns = df.columns.droplevel(level = 0)

df.drop(['Rk','Date','Year','Time','LTime','G#','Day','Result','OT','Unnamed: 6_level_1','PF','PD','PC','TD','XPA','XPM','FGA','FGM','2PA','2PM','Cmp','Att','Cmp%','Yds','TD','Yds.1','Rate','Tot'], **defColumnSettings)
df = df[df['Tm'] != 'Tm']

df.fillna(0, inplace=True)

for x in i:
    try:
        response = s.get(x)
        soup = BeautifulSoup(response.content, 'html.parser')    
        table = soup.find('table', {'id': 'results'})
        cf = pd.read_html(str(table))[0]
        cf.columns = cf.columns.droplevel(level = 0)
        cf.drop(['Rk','Date','Year','Time','LTime','G#','Day','Result','OT','Unnamed: 6_level_1','PF','PD','PC','TD','XPA','XPM','FGA','FGM','2PA','2PM','Cmp','Att','Cmp%','Yds','TD','Yds.1','Rate','Tot'], **defColumnSettings)
        cf = cf[cf['Tm'] != 'Tm']
        cf.fillna(0, inplace=True)
        df=pd.concat([df,cf])
    except:
        pass
    
Tm=df['Tm']
df.drop(labels=['Tm'],axis=1,inplace=True)
df.insert(0,'Tm',Tm)

Week=df['Week']
df.drop(labels=['Week'],axis=1,inplace=True)
df.insert(1,'Week',Week)


##defense conditions
URLA = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max={year}&order_by_asc=0&week_num_min={weekmin}&game_type=R&game_num_min=0&year_min={year}&temperature=10&offset=0
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URLB = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max={year}&order_by_asc=0&week_num_min={weekmin}&game_type=R&game_num_min=0&year_min={year}&temperature=10&offset=100
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URLC = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max={year}&order_by_asc=0&week_num_min={weekmin}&game_type=R&game_num_min=0&year_min={year}&temperature=10&offset=200
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URLD = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max={year}&order_by_asc=0&week_num_min={weekmin}&game_type=R&game_num_min=0&year_min={year}&temperature=10&offset=300
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

URLE = """
https://stathead.com/football/tgl_finder.cgi?request=1&temperature_gtlt=gt&game_num_max=99&week_num_max={weekmax}&order_by=game_date&match=game&year_max={year}&order_by_asc=0&week_num_min={weekmin}&game_type=R&game_num_min=0&year_min={year}&temperature=10&offset=400
""".format(year=year, weekmin=weekmin, weekmax=weekmax)

j = [URLB,URLC,URLD,URLE]

response = s.get(URLA)
soup = BeautifulSoup(response.content, 'html.parser')    
table = soup.find('table', {'id': 'results'})
gf = pd.read_html(str(table))[0]
gf.columns = gf.columns.droplevel(level = 0)
gf.drop(['Rk','Year','Date','Time','G#','Result','OT','Day','Opp'], **defColumnSettings)
gf = gf[gf['LTime'] != 'LTime']

for x in j:
    try:
        response = s.get(x)
        soup = BeautifulSoup(response.content, 'html.parser')    
        table = soup.find('table', {'id': 'results'})
        hf = pd.read_html(str(table))[0]
        hf.columns = hf.columns.droplevel(level = 0)
        hf.drop(['Rk','Year','Date','Time','G#','Result','OT','Day','Opp'], **defColumnSettings)
        hf = hf[hf['LTime'] != 'LTime']
        gf=pd.concat([gf,hf])
    except:
        pass

gf.rename({'Unnamed: 6_level_1':'SYMB'}, **defColumnSettings)
gf['HomeAway'] = np.where(gf['SYMB']!= '@','Home','Away')
gf =gf.drop(['SYMB'],axis=1)
gcombinedf = df.merge(gf, how='inner', left_on=['Tm','Week'], right_on=['Tm','Week'])

gcombinedf[['Pos']] = 'DST'

##Create formatted defense data
print("Formatting Defense Data for Model Training...")
OverallPA, OverallSfty, OverallTO, OverallKRTD, OverallPRTD, OverallIRTD, OverallFRTD, OverallOR, OverallSk = reprocess_defenses(gcombinedf,window)

##Create formatted player data
print("Formatting Player Data for Model Training...")
combinedf = combinedf.dropna()
combinedf = combinedf[combinedf['Tm'] != 'Pos']
OverallRecTD, OverallRec, OverallRecYds, OverallRushTD, OverallRushYds, OverallPassTD, OverallPassYds, OverallFR, OverallFL, OverallPCD, OverallINT = reprocess_players(combinedf,window)


##start model work
print("Training Models...")
print('PCD')
x, y = encode(OverallPCD,window)
model_PCD, scores_PCD = rrmodel(x,y)
print(model_PCD.coef_)

print('FL')
x, y = encode(OverallFL,window)
model_FL, scores_FL = rrmodel(x,y)
print(model_FL.coef_)

print('FR')
x, y = encode(OverallFR,window)
model_FR, scores_FR = rrmodel(x,y)
print(model_FR.coef_)

print('INT')
x, y = encode(OverallINT,window)
model_INT, scores_INT = rrmodel(x,y)
print(model_INT.coef_)

print('PassTD')
x, y = encode(OverallPassTD,window)
model_PassTD, scores_PassTD = rrmodel(x,y)
print(model_PassTD.coef_)

print('PassYds')
x, y = encode(OverallPassYds,window)
model_PassYds, scores_PassYds = rrmodel(x,y)
print(model_PassYds.coef_)

print('Rec')
x, y = encode(OverallRec,window)
model_Rec, scores_Rec = rrmodel(x,y)
print(model_Rec.coef_)

print('RecTD')
x, y = encode(OverallRecTD,window)
model_RecTD, scores_RecTD = rrmodel(x,y)
print(model_RecTD.coef_)

print('RecYds')
x, y = encode(OverallRecYds,window)
model_RecYds, scores_RecYds = rrmodel(x,y)
print(model_RecYds.coef_)

print('RushTD')
x, y = encode(OverallRushTD,window)
model_RushTD, scores_RushTD = rrmodel(x,y)
print(model_RushTD.coef_)

print('RushYds')
x, y = encode(OverallRushYds,window)
model_RushYds, scores_RushYds = rrmodel(x,y)
print(model_RushYds.coef_)

print('TO')
x, y = encode(OverallTO,window)
model_TO, scores_TO = rrmodel(x,y)
print(model_TO.coef_)

print('FR')
x, y = encode(OverallFRTD,window)
model_FR, scores_FR = rrmodel(x,y)
print(model_FR.coef_)

print('IR')
x, y = encode(OverallIRTD,window)
model_IR, scores_IR = rrmodel(x,y)
print(model_IR.coef_)

print('KR')
x, y = encode(OverallKRTD,window)
model_KR, scores_KR = rrmodel(x,y)
print(model_KR.coef_)

print('OR')
x, y = encode(OverallOR,window)
model_OR, scores_OR = rrmodel(x,y)
print(model_OR.coef_)

print('PA')
x, y = encode(OverallPA,window)
model_PA, scores_PA = rrmodel(x,y)
print(model_PA.coef_)

print('PR')
x, y = encode(OverallPRTD,window)
model_PR, scores_PR = rrmodel(x,y)
print(model_PR.coef_)

print('Sk')
x, y = encode(OverallSk,window)
model_Sk, scores_Sk = rrmodel(x,y)
print(model_Sk.coef_)

print('Sfty')
x, y = encode(OverallSfty,window)
model_Sfty, scores_Sfty = rrmodel(x,y)
print(model_Sfty.coef_)


models = {'PCD': model_PCD,
          'FL': model_FL,
          'FR': model_FR,
          'INT': model_INT,
          'PassTD': model_PassTD,
          'PassYds': model_PassYds,
          'Rec': model_Rec,
          'RecTD': model_RecTD,
          'RecYds': model_RecYds,
          'RushTD': model_RushTD,
          'RushYds': model_RushYds}

models_defense = {'TO': model_TO,
          'FR': model_FR,
          'IR': model_IR,
          'KR': model_KR,
          'OR': model_OR,
          'PA': model_PA,
          'PR': model_PR,
          'Sk': model_Sk,
          'Sfty': model_Sfty}

gh = get_recent_player_history(combinedf,window)
player_FP = expected_FP(combinedf, gh,window)

# write results to csv
df = pd.DataFrame(player_FP)
playerf = df.T

gh = get_recent_defense_history(gcombinedf,window) 
defense_FP = expected_defense_FP(gcombinedf, gh,window)

# write results to csv
df = pd.DataFrame(defense_FP) 
defensef = df.T
defensef = defensef.reset_index(level=0,inplace=False)
defensef = defensef.replace(to_replace="ARI",value="Arizona, Cardinals")
defensef = defensef.replace(to_replace="ATL",value="Atlanta, Falcons")
defensef = defensef.replace(to_replace="BAL",value="Baltimore, Ravens")
defensef = defensef.replace(to_replace="BUF",value="Buffalo,Bills")
defensef = defensef.replace(to_replace="CAR",value="Carolina,Panthes")
defensef = defensef.replace(to_replace="CHI",value="Chicago, Bears")
defensef = defensef.replace(to_replace="CIN",value="Cincinnati, Bengals")
defensef = defensef.replace(to_replace="CLE",value="Cleveland, Browns")
defensef = defensef.replace(to_replace="DAL",value="Dallas, Cowboys")
defensef = defensef.replace(to_replace="DEN",value="Denver, Broncos")
defensef = defensef.replace(to_replace="DET",value="Detroit, Lions")
defensef = defensef.replace(to_replace="GNB",value="Green Bay, Packers")
defensef = defensef.replace(to_replace="HOU",value="Houston, Texans")
defensef = defensef.replace(to_replace="IND",value="Indianapolis, Colts")
defensef = defensef.replace(to_replace="JAX",value="Jacksonville, Jaguars")
defensef = defensef.replace(to_replace="KAN",value="Kansas City, Chiefs")
defensef = defensef.replace(to_replace="LAC",value="Los Angeles, Chargers")
defensef = defensef.replace(to_replace="LAR",value="Los Angeles, Rams")
defensef = defensef.replace(to_replace="LVR",value="Las Vegas, Raiders")
defensef = defensef.replace(to_replace="MIA",value="Miami, Dolphins")
defensef = defensef.replace(to_replace="MIN",value="Minnesota, Vikings")
defensef = defensef.replace(to_replace="NOR",value="New Orleans, Saints")
defensef = defensef.replace(to_replace="NWE",value="New England, Patriots")
defensef = defensef.replace(to_replace="NYG",value="New York, Giants")
defensef = defensef.replace(to_replace="NYJ",value="New York, Jets")
defensef = defensef.replace(to_replace="PHI",value="Philadelphia, Eagles")
defensef = defensef.replace(to_replace="PIT",value="Pittsburgh, Steelers")
defensef = defensef.replace(to_replace="SEA",value="Seattle, Seahawks")
defensef = defensef.replace(to_replace="SFO",value="San Francisco, 49ers")
defensef = defensef.replace(to_replace="TAM",value="Tampa Bay, Buccaneers")
defensef = defensef.replace(to_replace="TEN",value="Tennessee, Titans")


defensef = defensef.rename({'index': 'Player'},axis=1,inplace=False)
dataIndef = defense_salary.merge(defensef, how='inner', left_on=['Player'], right_on=['Player'])
dataIndef = dataIndef.drop(columns = 'Pos_y',axis=0,inplace=False)
dataIndef = dataIndef.rename({'Pos_x': 'Pos'},axis=1,inplace=False)
dataIndef = dataIndef.rename({'SALARY': 'Cost'},axis=1,inplace=False)
dataIndef = dataIndef.rename({'Expected': 'Fan Points'},axis=1,inplace=False)

playerf = playerf.reset_index(level=0,inplace=False)
playerf = playerf.rename({'index': 'Player'},axis=1,inplace=False)
dataInpla = player_salary.merge(playerf, how='inner', left_on=['Player'], right_on=['Player'])
dataInpla = dataInpla.drop(columns = 'Pos_y',axis=0,inplace=False)
dataInpla = dataInpla.rename({'Pos_x': 'Pos', 'SALARY':'Cost','Expected': 'Fan Points'},axis=1,inplace=False)
outputOpt=pd.concat([dataInpla,dataIndef],ignore_index=True)
outputOpt[['$','Cost']]=outputOpt.Cost.str.split(pat='$',expand=True)
outputOpt = outputOpt.drop(columns = '$',axis=0,inplace=False)
outputOpt['Cost'] = outputOpt['Cost'].astype(int)

outputOpt=outputOpt.dropna()
optimize(outputOpt, positions = ['QB', 'RB', 'WR', 'TE', 'FLX', 'DST'])

outputDK = outputOpt
outputDK['Fan Points'] = outputDK['Cost'].div(300)
print("DK Optimization")
optimize(outputDK, positions = ['QB', 'RB', 'WR', 'TE', 'FLX', 'DST'])
