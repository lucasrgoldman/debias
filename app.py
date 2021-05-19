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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import io
import base64


app = Flask(__name__) 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
db = SQLAlchemy(app) # Linking DB


teams = Teams(year= '2020')
orig_df = teams.dataframes
orig_df.set_index('name', inplace=True)
orig_df.drop(['first_downs', 'first_downs_from_penalties',  'games_played','losses', 'abbreviation','pass_attempts', 'pass_completions', 'pass_first_downs','plays', 'points_contributed_by_offense','post_season_result', 'rush_attempts', 'rush_first_downs', 'wins'], axis=1, inplace= True)

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

    #@staticmethod
    def probToMoneyLine (self,prob):
        ml = 0
        prob*=100
        if prob >50:
            ml = -(prob/(100 - prob)) * 100
        elif prob < 50:
            ml = (((100 - prob)/prob) * 100)
        else:
            ml = 100
        ml = round(ml,2)
        mlStr = str(ml)
        if ml > 0:
            mlStr = "+" + mlStr
        return mlStr

    def predict (self, team1, team2):
        teams_df=self.getDF()
        t1 = teams_df['percentile'].loc[team1]
        t2 = teams_df['percentile'].loc[team2]
        p = 1/(10**(-(t1 - t2))+1)
        return self.probToMoneyLine(p)


    def plot(self):
        sns_plot = sns.kdeplot(self.teams_df['sum'])
        sns_plot.savefig("output.png")

    def predictGames(self):
        games = getML()
        for game in games:
            team1 = game['teams'][0]
            team2 = game['teams'][1]
            # print(team1, ' vs', team2)
            return 'Predicted Line for', team1,'is', predict(team1, team2)
            # print('Actual Line for', team1,'is', game['odds'][0] )

    def getDF(self):
        teams_df = orig_df
        for (columnName, columnData) in self.teams_df.iteritems(): 
            if columnName != 'name':
                teams_df[columnName] = stats.zscore(columnData)
        teams_df['fumbles'] *= -1
        teams_df['interceptions'] *= -1
        teams_df['penalties'] *= -1
        teams_df['percent_drives_with_turnovers'] *= -1
        teams_df['points_against'] *= -1
        teams_df['turnovers'] *= -1
        teams_df['yards_from_penalties'] *= -1
        rank = pd.Series()
        rank['defensive_simple_rating_system'] = self.defensive_simple_rating_system
        rank['fumbles'] =self.fumbles
        rank['interceptions'] =self.interceptions
        rank['margin_of_victory'] = self.margin_of_victory
        rank['offensive_simple_rating_system'] = self.offensive_simple_rating_system
        rank['pass_net_yards_per_attempt'] = self.pass_net_yards_per_attempt
        rank['pass_touchdowns'] = self.pass_touchdowns
        rank['pass_yards'] =self.pass_yards
        rank['penalties'] =self.penalties
        rank['percent_drives_with_points'] =self.percent_drives_with_points
        rank['percent_drives_with_turnovers'] = self.percent_drives_with_turnovers
        rank['points_against'] =self.points_against
        rank['rank'] = self.rank
        rank['rush_touchdowns'] = self.rush_touchdowns
        rank['rush_yards'] = self.rush_yards
        rank['rush_yards_per_attempt'] =self.rush_yards_per_attempt
        rank['simple_rating_system'] = self.simple_rating_system
        rank['strength_of_schedule'] = self.strength_of_schedule
        rank['turnovers'] = self.turnovers
        rank['win_percentage'] = self.win_percentage
        rank['yards'] = self.yards
        rank['yards_from_penalties'] = self.yards_from_penalties
        rank['yards_per_play'] = self.yards_per_play
        sum = rank.sum() 
        rank/=sum
        for (columnName, columnData) in rank.iteritems(): 
            teams_df[columnName]*= columnData
        teams_df['sum'] = 0.0
        for i, row in teams_df.iterrows():
            teams_df.at[i, 'sum'] = row['defensive_simple_rating_system':].sum()
        teams_df.sort_values(by=['sum'], inplace=True, ascending=False)
        teams_df['zscores'] = stats.zscore(teams_df['sum'])
        teams_df['percentile'] =  1- stats.norm.sf(teams_df['zscores'])
        return teams_df

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

@app.route('/prediction/predict/<int:id>', methods = ['GET','POST'])
def predict(id):
    pred = Prediction.query.get_or_404(id)
    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        ml = pred.predict(team1, team2)
        return render_template('predict.html',  pred=pred,team1=team1, team2=team2,ml=ml)
    else:
        return render_template('listteams.html', pred=pred)

@app.route('/prediction/view/<int:id>')
def view(id):
    pred = Prediction.query.get_or_404(id)
    view_df = pred.getDF()
    view_df = view_df.round(2)
    view_df = view_df[['sum', 'zscores', 'percentile']]
    img = io.BytesIO()
    sns_plot = sns.kdeplot(view_df['sum'])
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    # img_html = '<img src="data:image/png;base64,{}">'.format(plot_url)
    return render_template('view.html', plot_url=plot_url, tables=[view_df.to_html(classes='data')], titles=view_df.columns.values)

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
