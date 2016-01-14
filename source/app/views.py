from app import app
from flask import render_template
from flask import request
from flask import Flask, session
from main import processFacilities, getReview

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

def extractTopicsWords(twlist):
	#Getting topic words
	temp = []
	tWrodsL = twlist.split('+')
	for words in tWrodsL:
		word = words.split('*')
		temp.append(word[1])
	return ', '.join(temp)

def analysisBusiness(minrl,maxrl,topfaci,topicsSize,season,topicLen=5):
	faci,reviewSelected,topics = processFacilities(minrl,maxrl,topicsSize,topicLen,season)
	faci = faci.items()
	facilities = []
	topfaci = int(topfaci)
	count = 1
	for item in faci:
		if topfaci >= count:
			facilities.append(list(item))
			count += 1
		else:
			break
	print facilities

	#process topics
	topicsList = []
	for topic in topics:
		temp = []
		temp.append('Topic '+str(topic[0]))
		#Process topic[1] - get words list
		temp.append(extractTopicsWords(topic[1]))
		topicsList.append(temp)

	return facilities,reviewSelected,topicsList

@app.route('/getrating/',methods=['POST'])
def getRating():
	if request.method == 'POST':
		minrl = session['minrl']
		maxrl = session['maxrl']
		seas = session['seas']
		topics = session['topics']
		facilities = session['facilities']
		reviewSelected = session['reviewSelected']
		rText = request.form['rText']
		rating = getReview(rText)	
		return render_template('analysis.html',topics=topics,facilities = facilities,reviewSelected=reviewSelected,minrl=minrl,maxrl=maxrl,seas=seas,rating=rating,rText=rText)

@app.route('/getinsight/',methods=['POST'])
def analysisMain():
	if request.method == 'POST':
		minrl = request.form['minrl']
		maxrl = request.form['maxrl']
		topfaci = request.form['faci']
		topicsSize = request.form['topics']
		season = request.form['seas']
		print minrl, maxrl, topfaci,topicsSize,season
		#Get insights
		facilities,reviewSelected,topics = analysisBusiness(minrl,maxrl,topfaci,topicsSize,season)
		if season == '0' :
			seas = 'All'
		elif season == '1':
			seas = 'Spring'
		elif season == '2':
			seas = 'Summer'
		elif season == '3':
			seas = 'Autumn'
		else:
			seas = 'Winter'

		rating = 'Null'
		rText = ''

		session['minrl']= minrl
		session['maxrl'] = maxrl
		session['seas'] = seas
		session['topics'] = topics
		session['facilities'] = facilities
		session['reviewSelected'] = reviewSelected

		return render_template('analysis.html',topics=topics,facilities = facilities,reviewSelected=reviewSelected,minrl=minrl,maxrl=maxrl,seas=seas,rating=rating,rText=rText)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/analysis')
def analysis():
	df = [['Topics','N/A']]
	topics = df
	facilities = [['Null','0.0']]
	reviewSelected = '0'
	maxrl = 0
	minrl = 0
	seas = 'Null'
	rating = 'Null'
	rText=''
	return render_template('analysis.html',topics=topics,facilities=facilities,reviewSelected=reviewSelected,minrl=minrl,maxrl=maxrl,seas =seas,rating=rating,rText=rText )

