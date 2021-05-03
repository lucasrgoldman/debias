from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sportsreference.nfl.teams import Teams
import pandas as pd
import numpy as np
from scipy import stats
import json
import requests
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__) 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
db = SQLAlchemy(app) # Linking DB

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    defensive_simple_rating_system = db.Column(db.Integer, nullable=False)
    fumbles = db.Column(db.Integer, nullable=False)
    interceptions = db.Column(db.Integer, nullable=False)
    margin_of_victory = db.Column(db.Integer, nullable=False)
    offensive_simple_rating_system = db.Column(db.Integer, nullable=False)
    pass_net_yards_per_attempt = db.Column(db.Integer, nullable=False)
    pass_touchdowns = db.Column(db.Integer, nullable=False)
    pass_yards = db.Column(db.Integer, nullable=False)
    penalties = db.Column(db.Integer, nullable=False)
    percent_drives_with_points = db.Column(db.Integer, nullable=False)
    percent_drives_with_turnovers = db.Column(db.Integer, nullable=False)
    points_against = db.Column(db.Integer, nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    rush_touchdowns = db.Column(db.Integer, nullable=False)
    rush_yards = db.Column(db.Integer, nullable=False)
    rush_yards_per_attempt = db.Column(db.Integer, nullable=False)
    simple_rating_system = db.Column(db.Integer, nullable=False)
    strength_of_schedule = db.Column(db.Integer, nullable=False)
    turnovers = db.Column(db.Integer, nullable=False)
    win_percentage = db.Column(db.Integer, nullable=False)
    yards = db.Column(db.Integer, nullable=False)
    yards_from_penalties = db.Column(db.Integer, nullable=False)
    yards_per_play = db.Column(db.Integer, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    teams_df = pd.DataFrame()

    def __init__(self, defensive_simple_rating_system, fumbles, interceptions, margin_of_victory, offensive_simple_rating_system,pass_net_yards_per_attempt,pass_touchdowns,pass_yards,penalties, percent_drives_with_points, percent_drives_with_turnovers, points_against,rank, rush_touchdowns, rush_yards, rush_yards_per_attempt, simple_rating_system, strength_of_schedule, turnovers,win_percentage, yards, yards_from_penalties, yards_per_play):
        teams = Teams(year= '2020')
        self.teams_df = teams.dataframes
        self.teams_df.set_index('name', inplace=True)
        self.teams_df.drop(['first_downs', 'first_downs_from_penalties',  'games_played','losses', 'abbreviation','pass_attempts', 'pass_completions', 'pass_first_downs','plays', 'points_contributed_by_offense','post_season_result', 'rush_attempts', 'rush_first_downs', 'wins'], axis=1, inplace= True)
        for (columnName, columnData) in self.teams_df.iteritems(): 
            if columnName != 'name':
                self.teams_df[columnName] = stats.zscore(columnData)
        self.teams_df['fumbles'] *= -1
        self.teams_df['interceptions'] *= -1
        self.teams_df['penalties'] *= -1
        self.teams_df['percent_drives_with_turnovers'] *= -1
        self.teams_df['points_against'] *= -1
        self.teams_df['turnovers'] *= -1
        self.teams_df['yards_from_penalties'] *= -1
        self.rank = pd.Series()
        self.rank['defensive_simple_rating_system'] = defensive_simple_rating_system
        self.rank['fumbles'] =fumbles
        self.rank['interceptions'] =interceptions
        self.rank['margin_of_victory'] = margin_of_victory
        self.rank['offensive_simple_rating_system'] = offensive_simple_rating_system
        self.rank['pass_net_yards_per_attempt'] = pass_net_yards_per_attempt
        self.rank['pass_touchdowns'] = pass_touchdowns
        self.rank['pass_yards'] =pass_yards
        self.rank['penalties'] =penalties
        self.rank['percent_drives_with_points'] =percent_drives_with_points
        self.rank['percent_drives_with_turnovers'] = percent_drives_with_turnovers
        self.rank['points_against'] =points_against
        self.rank['rank'] = rank
        self.rank['rush_touchdowns'] = rush_touchdowns
        self.rank['rush_yards'] = rush_yards
        self.rank['rush_yards_per_attempt'] =rush_yards_per_attempt
        self.rank['simple_rating_system'] = simple_rating_system
        self.rank['strength_of_schedule'] = strength_of_schedule
        self.rank['turnovers'] = turnovers
        self.rank['win_percentage'] = win_percentage
        self.rank['yards'] = yards
        self.rank['yards_from_penalties'] = yards_from_penalties
        self.rank['yards_per_play'] = yards_per_play
        sum = self.rank.sum() 
        self.rank/=sum
        self.defensive_simple_rating_system = defensive_simple_rating_system
        self.fumbles = fumbles
        self.interceptions = interceptions
        self.margin_of_victory = margin_of_victory
        self.offensive_simple_rating_system = offensive_simple_rating_system
        self.pass_net_yards_per_attempt = pass_net_yards_per_attempt
        self.pass_touchdowns = pass_touchdowns
        self.pass_yards = pass_yards
        self.penalties = penalties
        self.percent_drives_with_points = percent_drives_with_points
        self.percent_drives_with_turnovers = percent_drives_with_turnovers
        self.points_against = points_against
        self.rank = rank
        self.rush_touchdowns = rush_touchdowns
        self.rush_yards = rush_yards
        self.rush_yards_per_attempt = rush_yards_per_attempt
        self.simple_rating_system = simple_rating_system
        self.strength_of_schedule = strength_of_schedule
        self.turnovers = turnovers
        self.win_percentage = win_percentage
        self.yards = yards
        self.yards_from_penalties = yards_from_penalties
        self.yards_per_play = yards_per_play
        for (columnName, columnData) in self.rank.iteritems(): 
            self.teams_df[columnName]*= columnData
        self.teams_df['sum'] = 0.0
        for i, row in self.teams_df.iterrows():
            self.teams_df.at[i, 'sum'] = row['defensive_simple_rating_system':].sum()
        self.teams_df.sort_values(by=['sum'], inplace=True, ascending=False)
        self.teams_df['zscores'] = stats.zscore(teams_df['sum'])
        self.teams_df['percentile'] =  1- stats.norm.sf(teams_df['zscores'])

    
    @staticmethod
    def predict (team1, team2):
        t1 = teams_df['percentile'].loc[team1]
        t2 = teams_df['percentile'].loc[team2]
        p = 1/(10**(-(t1 - t2))+1)
        return probToMoneyLine(p)

    @staticmethod
    def probToMoneyLine (prob):
        ml = 0
        prob*=100
        if prob >50:
            ml = -(prob/(100 - prob)) * 100
        elif prob < 50:
            ml = (((100 - prob)/prob) * 100)
        else:
            ml = 100
        return ml

    def plot(self):
        sns.kdeplot(self.teams_df['sum'])

    def predictGames(self):
        games = getML()
        for game in games:
            team1 = game['teams'][0]
            team2 = game['teams'][1]
            # print(team1, ' vs', team2)
            # print('Predicted Line for', team1,'is', predict(team1, team2))
            # print('Actual Line for', team1,'is', game['odds'][0] )

    def to_html(self):
        return teams_df['percentile'].to_html

    @staticmethod
    def getML():
        parser = argparse.ArgumentParser(description='Sample')
        parser.add_argument('--api-key', type=str, default='')
        args, unknown = parser.parse_known_args()
        API_KEY = '58f860df380e5b01f108f9418584b714'
        SPORT = 'americanfootball_nfl' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports
        REGION = 'us' # uk | us | eu | au
        MARKET = 'h2h' # h2h | spreads | totals
        ODDSFORMAT  = 'american'
    
        # Now get a list of live & upcoming games for the sport you want, along with odds for different bookmakers
    
        odds_response = requests.get('https://api.the-odds-api.com/v3/odds', params={
            'api_key': API_KEY,
            'sport': SPORT,
            'region': REGION,
            'mkt': MARKET,
            'oddsFormat': ODDSFORMAT,
        })

        odds_json = json.loads(odds_response.text)
        games = []
        if not odds_json['success']:
            print(odds_json['msg'])
        else:
            print('Number of events:', len(odds_json['data']))
            # print(odds_json['data'])
            for i, game in enumerate(odds_json['data'], start=0):
                games.append({})
                games[i]['teams'] = game['teams']
                games[i]['home'] = game['home_team']

                for site in game['sites']:
                    if site['site_nice'] == 'Caesars':
                        games[i]['odds'] = site['odds']['h2h']

            # Check your usage
            # print('Remaining requests', odds_response.headers['x-requests-remaining'])
            # print('Used requests', odds_response.headers['x-requests-used'])

        return games

    def __repr__(self):
        return 'Prediction  Number: ' + str(self.id)
        

@app.route('/') #base url
def index():
    return render_template('index.html')
    
@app.route('/prediction')
def prediction():
    all_preds = Prediction.query.order_by(Prediction.date_posted).all()
    return render_template('prediction.html', preds=all_preds)

@app.route('/prediction/delete/<int:id>')
def delete(id):
    pred = Prediction.query.get_or_404(id)
    db.session.delete(pred)
    db.session.commit()
    return redirect('/prediction')

@app.route('/prediction/edit/<int:id>', methods = ['GET','POST'])
def edit(id):
    pred = Prediction.query.get_or_404(id)
    if request.method == 'POST':
        pred.defensive_simple_rating_system = request.form['defensive_simple_rating_system']
        pred.fumbles = request.form['fumbles']
        pred.interceptions = request.form['interceptions']
        pred.margin_of_victory = request.form['margin_of_victory']
        pred.offensive_simple_rating_system = request.form['offensive_simple_rating_system']
        pred.pass_net_yards_per_attempt = request.form['pass_net_yards_per_attempt']
        pred.pass_touchdowns = request.form['pass_touchdowns']
        pred.pass_yards = request.form['pass_yards']
        pred.penalties = request.form['penalties']
        pred.percent_drives_with_points = request.form['percent_drives_with_points']
        pred.percent_drives_with_turnovers = request.form['percent_drives_with_turnovers']
        pred.points_against = request.form['points_against']
        pred.rank = request.form['rank'] 
        pred.rush_touchdowns = request.form['rush_touchdowns']
        pred.rush_yards = request.form['rush_yards']
        pred.rush_yards_per_attempt = request.form['rush_yards_per_attempt']
        pred.simple_rating_system = request.form['simple_rating_system']
        pred.strength_of_schedule = request.form['strength_of_schedule']
        pred.turnovers = request.form['turnovers']
        pred.win_percentage = request.form['win_percentage']
        pred.yards = request.form['yards']
        pred.yards_from_penalties = request.form['yards_from_penalties']
        pred.yards_per_play = request.form['yards_per_play']
        db.session.commit()
        return redirect('/prediction')
    else:
        return render_template('edit.html', pred=pred)

@app.route('/prediction/new', methods=['GET', 'POST'])
def new_prediction():
    if request.method == 'POST':
        defensive_simple_rating_system = request.form['defensive_simple_rating_system']
        fumbles = request.form['fumbles']
        interceptions = request.form['interceptions']
        margin_of_victory = request.form['margin_of_victory']
        offensive_simple_rating_system = request.form['offensive_simple_rating_system']
        pass_net_yards_per_attempt = request.form['pass_net_yards_per_attempt']
        pass_touchdowns = request.form['pass_touchdowns']
        pass_yards = request.form['pass_yards']
        penalties = request.form['penalties']
        percent_drives_with_points = request.form['percent_drives_with_points']
        percent_drives_with_turnovers = request.form['percent_drives_with_turnovers']
        points_against = request.form['points_against']
        rank = request.form['rank'] 
        rush_touchdowns = request.form['rush_touchdowns']
        rush_yards = request.form['rush_yards']
        rush_yards_per_attempt = request.form['rush_yards_per_attempt']
        simple_rating_system = request.form['simple_rating_system']
        strength_of_schedule = request.form['strength_of_schedule']
        turnovers = request.form['turnovers']
        win_percentage = request.form['win_percentage']
        yards = request.form['yards']
        yards_from_penalties = request.form['yards_from_penalties']
        yards_per_play = request.form['yards_per_play']
        new_pred= Prediction(int(defensive_simple_rating_system), int(fumbles), int(interceptions), int(margin_of_victory), int(offensive_simple_rating_system),int(pass_net_yards_per_attempt),int(pass_touchdowns),int(pass_yards),int(penalties), int(percent_drives_with_points), int(percent_drives_with_turnovers), int(points_against),int(rank), int(rush_touchdowns), int(rush_yards), int(rush_yards_per_attempt), int(simple_rating_system), int(strength_of_schedule), int(turnovers),int(win_percentage), int(yards), int(yards_from_penalties), int(yards_per_play))
        db.session.add(new_pred)
        db.session.commit()
        return redirect('/prediction')
    else:
        return render_template('new_prediction.html') # send var to posts.htmls

if __name__ == "__main__": #if from terminal
    app.run(debug=True)
