from firebase import firebase
firebase = firebase.FirebaseApplication('https://project-fyp-161f7-default-rtdb.firebaseio.com/')

def post_data(A,B,D,E):

	data={'A':A,'B':B,'D':D,'E':E}
	result= firebase.post('/project-fyp-161f7-default-rtdb/', data)


			
