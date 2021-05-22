# See nfl-predictor.ipynb for documentation

from app import app
from datetime import datetime
import base64
import io
import seaborn as sns
from matplotlib import pyplot as plt
from flask import Flask, render_template, request, redirect
from sportsreference.nfl.teams import Teams
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from scipy import stats
import json
import requests
import argparse
import matplotlib
matplotlib.use('Agg')

db = SQLAlchemy(app)  # Linking DB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# orig_df is a global so sports-reference only fetches data once

teams = Teams(year='2020')
orig_df = teams.dataframes
orig_df.set_index('name', inplace=True)
orig_df.drop(
    ['first_downs', 'first_downs_from_penalties', 'games_played', 'losses', 'abbreviation', 'pass_attempts',
     'pass_completions', 'pass_first_downs', 'plays', 'points_contributed_by_offense', 'post_season_result',
     'rush_attempts', 'rush_first_downs', 'wins', 'points_for','points_against'],
    axis=1, inplace=True)
for (columnName, columnData) in orig_df.iteritems():
    if columnName != 'name':
        orig_df[columnName] = stats.zscore(columnData)
orig_df[['fumbles', 'interceptions', 'penalties', 'percent_drives_with_turnovers',
         'turnovers', 'yards_from_penalties']] *= -1


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
    points_difference = db.Column(db.Integer, nullable=False)
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
    date_posted = db.Column(db.DateTime, nullable=False,
                            default=datetime.utcnow)

    def __repr__(self):
        return 'Prediction  Number: ' + str(self.id)


def getMoneyLine():
    parser = argparse.ArgumentParser(description='Sample')
    parser.add_argument('--api-key', type=str, default='')
    args, unknown = parser.parse_known_args()
    API_KEY = '58f860df380e5b01f108f9418584b714'
    # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports
    SPORT = 'americanfootball_nfl'
    REGION = 'us'  # uk | us | eu | au
    MARKET = 'h2h'  # h2h | spreads | totals
    ODDSFORMAT = 'american'

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
                if site['site_nice'] == 'DraftKings':
                    if site['odds']['h2h'][0] > 0:
                        games[i]['odds'] = '+' + str(site['odds']['h2h'][0])
    return games


def probToMoneyLine(prob):
    ml = 0
    prob *= 100
    if prob > 50:
        ml = -(prob/(100 - prob)) * 100
    elif prob < 50:
        ml = (((100 - prob)/prob) * 100)
    else:
        ml = 100
    ml = round(ml, 2)
    mlStr = str(ml)
    if ml > 0:
        mlStr = "+" + mlStr
    return mlStr


def predictMoneyLine(pred, team1, team2):
    teams_df = getDF(pred)
    t1 = teams_df['percentile'].loc[team1]
    t2 = teams_df['percentile'].loc[team2]
    p = 1/(10**(-(t1 - t2))+1)
    return probToMoneyLine(p)


def predictGames(pred):
    games = getML(pred)
    for game in games:
        team1 = game['teams'][0]
        team2 = game['teams'][1]
        return 'Predicted Line for', team1, 'is', predict(team1, team2)


def getDF(pred):
    teams_df = orig_df.copy()
    print(orig_df)
    weights = pd.Series()
    weights['defensive_simple_rating_system'] = pred.defensive_simple_rating_system
    weights['fumbles'] = pred.fumbles
    weights['interceptions'] = pred.interceptions
    weights['margin_of_victory'] = pred.margin_of_victory
    weights['offensive_simple_rating_system'] = pred.offensive_simple_rating_system
    weights['pass_net_yards_per_attempt'] = pred.pass_net_yards_per_attempt
    weights['pass_touchdowns'] = pred.pass_touchdowns
    weights['pass_yards'] = pred.pass_yards
    weights['penalties'] = pred.penalties
    weights['percent_drives_with_points'] = pred.percent_drives_with_points
    weights['percent_drives_with_turnovers'] = pred.percent_drives_with_turnovers
    weights['points_difference'] = pred.points_difference
    weights['rank'] = pred.rank
    weights['rush_touchdowns'] = pred.rush_touchdowns
    weights['rush_yards'] = pred.rush_yards
    weights['rush_yards_per_attempt'] = pred.rush_yards_per_attempt
    weights['simple_rating_system'] = pred.simple_rating_system
    weights['strength_of_schedule'] = pred.strength_of_schedule
    weights['turnovers'] = pred.turnovers
    weights['win_percentage'] = pred.win_percentage
    weights['yards'] = pred.yards
    weights['yards_from_penalties'] = pred.yards_from_penalties
    weights['yards_per_play'] = pred.yards_per_play
    sums = weights.sum()
    print(sums)
    weights /= sums
    for (columnName, columnData) in weights.iteritems():
        teams_df[columnName] *= columnData
    teams_df['sum'] = 0.0
    for i, row in teams_df.iterrows():
        teams_df.at[i, 'sum'] = row['defensive_simple_rating_system':].sum()
    teams_df.sort_values(by=['sum'], inplace=True, ascending=False)
    teams_df['zscores'] = stats.zscore(teams_df['sum'])
    teams_df['percentile'] = 1 - stats.norm.sf(teams_df['zscores'])
    return teams_df


@app.route('/')  # base url
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


@app.route('/prediction/predict/<int:id>', methods=['GET', 'POST'])
def predict(id):
    pred = Prediction.query.get_or_404(id)
    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        ml = predictMoneyLine(pred, team1, team2)
        return render_template('predict.html',  pred=pred, team1=team1, team2=team2, ml=ml)
    else:
        return render_template('listteams.html', pred=pred)


@app.route('/prediction/predict/<int:id>/weekly_matchups')
def weekly_matchups(id):
    pred = Prediction.query.get_or_404(id)
    games = getMoneyLine()
    predictions = []
    i = 0
    for game in games:
        if i > 15:  # Only fetch one week of games
            break
        if 'odds' in game:
            team1 = game['teams'][0]
            team2 = game['teams'][1]
            predictions.append({})
            predictions[i]['team1'] = team1
            predictions[i]['team2'] = team2
            predictions[i]['prediction'] = predictMoneyLine(pred, team1, team2)
            predictions[i]['vegas_odds'] = game['odds']
            i += 1
    return render_template('weekly_matchups.html',  predictions=predictions)


@app.route('/prediction/view/<int:id>')
def view(id):
    pred = Prediction.query.get_or_404(id)
    view_df = getDF(pred)
    view_df = view_df.round(2)
    view_df = view_df[['sum', 'zscores', 'percentile']]
    view_df.reset_index(inplace=True)
    view_df.rename(columns={'name':'Team Name','sum': 'Cumulative Score',
                   'zscores': 'Z-Score', 'percentile': 'Percentile'}, inplace=True)
    view_df.index = np.arange(1, len(view_df)+1)
    view_df.index.name = 'Ranking'
    img = io.BytesIO()
    sns.set()
    sns_plot = sns.kdeplot(view_df['Cumulative Score']).set_title("Plot of Cumulative Scores")
    plt.savefig(img, format='png')
    plt.clf()
    plt.cla()
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('view.html', plot_url=plot_url, tables=[view_df.to_html(classes='data')])


@app.route('/prediction/edit/<int:id>', methods=['GET', 'POST'])
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
        pred.points_difference = request.form['points_difference']
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
        points_difference = request.form['points_difference']
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
        new_pred = Prediction(
            defensive_simple_rating_system=int(defensive_simple_rating_system),
            fumbles=int(fumbles),
            interceptions=int(interceptions),
            margin_of_victory=int(margin_of_victory),
            offensive_simple_rating_system=int(offensive_simple_rating_system),
            pass_net_yards_per_attempt=int(pass_net_yards_per_attempt),
            pass_touchdowns=int(pass_touchdowns),
            pass_yards=int(pass_yards),
            penalties=int(penalties),
            percent_drives_with_points=int(percent_drives_with_points),
            percent_drives_with_turnovers=int(percent_drives_with_turnovers),
            points_difference=int(points_difference),
            rank=int(rank),
            rush_touchdowns=int(rush_touchdowns),
            rush_yards=int(rush_yards),
            rush_yards_per_attempt=int(rush_yards_per_attempt),
            simple_rating_system=int(simple_rating_system),
            strength_of_schedule=int(strength_of_schedule),
            turnovers=int(turnovers),
            win_percentage=int(win_percentage),
            yards=int(yards),
            yards_from_penalties=int(yards_from_penalties),
            yards_per_play=int(yards_per_play))
        db.session.add(new_pred)
        db.session.commit()
        return redirect('/prediction')
    else:
        return render_template('new_prediction.html')
