import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import time
import sys
import csv
import pdb
from tqdm import tqdm
from os.path import join, dirname

# Need this for relative import
sys.path.append("..")

from src.utils.path_names import raw_episodes_path

"""
This is an edited version of David NQ's Halite Game Scraper at https://www.kaggle.com/david1013/halite-game-scraper
Kaggle's API limit for Google Football is yet to be made explicit but in Kaggle Halite the limit of 1000 requests per day was eventually raised to 3600 requests per day max.
Rate limits are shared between the ListEpisodes and GetEpisodeReplay endpoints. Exceeding limits repeatedly will lead to temporary and then permanent bans. At some point it is expected Kaggle will remove this public API and provide datasets of episodes.
The episodes take a lot of space. In Kaggle Halite, I ended up with 200GB of games. The Football JSON files are ten times larger so you may end up with terabytes. If you use this or any scraper, consider posting the dataset to Kaggle Datasets for others to use.
"""

MAX_EPISODES = 500
MIN_SCORE = 1500
NUM_TOP_TEAMS = 1

episode_dir = raw_episodes_path
print("Episode Directory: " + episode_dir)

if not os.path.exists(episode_dir):
    os.makedirs(episode_dir, exist_ok=True)

base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
list_url = base_url + "ListEpisodes"

def getTeamEpisodes(team_id):

    r = requests.post(list_url, json = {"teamId":  int(team_id)})
    rj = r.json()

    # update teams list
    #global teams_df
    #teams_df_new = pd.DataFrame(rj['result']['teams'])

    ##  print(teams_df_new)

    #if len(teams_df.columns) == len(teams_df_new.columns) and (teams_df.columns == teams_df_new.columns).all():
    #    teams_df = pd.concat( (teams_df, teams_df_new.loc[[c for c in teams_df_new.index if c not in teams_df.index]] ) )
    #    teams_df.sort_values('publicLeaderboardRank', inplace = True)
    #else:
    #    print('teams dataframe did not match')

    # make df
    team_episodes = pd.DataFrame(rj['result']['episodes'])

    #team_episodes['avg_score'] = -1;

    # print(team_episodes)

    for i in range(len(team_episodes)):
        agents = team_episodes['agents'].loc[i]
        agent_scores = [a['updatedScore'] for a in agents if a['updatedScore'] is not None]
        team_episodes.loc[i, 'submissionId'] = max([a['submissionId'] for a in agents])
        if len(agent_scores):
            team_episodes.loc[i, 'updatedScore'] = max([a['updatedScore'] for a in agents])
            scores = [a['updatedScore'] for a in agents]
            team_episodes.loc[i, 'left_team'] = scores[0] > scores[1]
        else:
            team_episodes.loc[i, 'updatedScore'] = -100

    team_episodes = team_episodes.loc[team_episodes['updatedScore'] > 0]

    team_episodes['final_score'] = team_episodes['updatedScore']
    team_episodes.sort_values('final_score', ascending = False, inplace=True)

    return rj, team_episodes

#def getTeamEpisodes(team_id):
#
#    r = requests.post(list_url, json = {"teamId":  int(team_id)})
#    rj = r.json()
#
#    teams_df = pd.DataFrame(rj['result']['teams'])
#    
#    # make df
#    team_episodes = pd.DataFrame(rj['result']['episodes'])
#    team_episodes['avg_score'] = -1;
#    
#    # Only get episodes data for team with id = team_id
#    for i in range(len(team_episodes)):
#        agents = team_episodes['agents'].loc[i]
#        agent_scores = [a['updatedScore'] for a in agents if a['updatedScore'] is not None]
#        team_episodes.loc[i, 'submissionId'] = [a['submissionId'] for a in agents if a['submission']['teamId'] == team_id][0]
#        team_episodes.loc[i, 'updatedScore'] = [a['updatedScore'] for a in agents if a['submission']['teamId'] == team_id][0]
#        
#        if len(agent_scores) > 0:
#            team_episodes.loc[i, 'avg_score'] = np.mean(agent_scores)
#
#    for sub_id in team_episodes['submissionId'].unique():
#        sub_rows = team_episodes[ team_episodes['submissionId'] == sub_id ]
#        max_time = max( [r['seconds'] for r in sub_rows['endTime']] )
#        final_score = max( [r['updatedScore'] for r_idx, (r_index, r) in enumerate(sub_rows.iterrows())
#                                if r['endTime']['seconds'] == max_time] )
#
#        team_episodes.loc[sub_rows.index, 'final_score'] = final_score
#        
#    team_episodes.sort_values('avg_score', ascending = False, inplace=True)
#    return rj, team_episodes

def saveEpisode(epid, rj):
    # request
    re = requests.post(get_url, json = {"EpisodeId": int(epid)})

    # save replay
    with open(episode_dir + '/{}.json'.format(epid), 'w') as f:
        f.write(re.json()['result']['replay'])

    # save episode info
    with open(episode_dir + '/{}_info.json'.format(epid), 'w') as f:
        json.dump([r for r in rj['result']['episodes'] if r['id']==epid][0], f)

episodes_count = 0
leaderboard_csv = join(dirname(episode_dir), "google-football-publicleaderboard.csv")

def download_episodes():

    leaderboard_df = pd.read_csv(leaderboard_csv)
    leaderboard_df.sort_values('Score', inplace = True, ascending = False)
    top_teams_df = leaderboard_df[:NUM_TOP_TEAMS]
    print(top_teams_df)
    top_teams_id = list(top_teams_df['TeamId'])
    top_teams_name = list(top_teams_df['TeamName'])
    
    episode_count = 0
    left_team = 0 # left team or right team

    # csv to keep track of episode score, team_name, team_id, left or right player
    metadata_csv_path = join(dirname(raw_episodes_path),"episodes_metadata.csv")
    print("Episodes Metadata Path: " + metadata_csv_path)

    with open(metadata_csv_path, "w") as metadata_csv:
        metadata_csv.write("epid,score,left_team,team_name,team_id\n")
        for team_id, team_name in zip(top_teams_id, top_teams_name):
            print("Parsing Team: " + str(team_name) + " (team id: " + str(team_id) + ")")
            team_json, team_df = getTeamEpisodes(team_id)
            epid_list = list(team_df['id'])
            score_list = list(team_df['final_score'])
            left_team_list = list(team_df['left_team'])
            for epid, score, if_left_team in zip(epid_list, score_list, left_team_list):
                if score > MIN_SCORE and episode_count < MAX_EPISODES:
                    # Only download if episode doesn't already exist
                    episode_path = join(episode_dir, str(epid) + '.json')
                    if not os.path.exists(episode_path):
                        saveEpisode(epid, team_json)
                        print(episode_path + " downloaded (score: " + str(score) + ") (Team: " + team_name + ")")
                    else:
                        print(episode_path + " already exists!")

                    # Extract left/right team info
                    with open(join(episode_dir,str(epid)+"_info.json")) as info_csv:
                        episode_info = json.load(info_csv)
                        #if episode_info["agents"][0]["submission"]["teamId"] == team_id:
                        #    left_team = 1
                        #else:
                        #    left_team = 0
                        if if_left_team:
                            left_team = 1

                    score = "%.2f"%(score) 
                    metadata_csv.write(str(epid) + "," + score + "," + str(left_team) + "," + team_name + "," + str(team_id) + "\n")

                    # Also count exiting episodes
                    episode_count += 1

if __name__ == "__main__":
    download_episodes()











