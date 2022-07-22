import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


kivy.require("1.10.1")


class LoginUI(GridLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.cols = 2
		#Window.size =(420,600)
		#self.add_widget(Label(text="Diabetes Checker Requirements"))

		self.add_widget(Label(text="Age:"))
		self.age = TextInput(multiline=False,write_tab=False)
		self.age.focus=True
		self.add_widget(self.age)

		self.add_widget(Label(text="Anaemia"))
		self.anaemia = TextInput(multiline=False,write_tab=False)
		self.anaemia.focus=True
		self.add_widget(self.anaemia)

		self.add_widget(Label(text="Creatinine"))
		self.creatinine = TextInput(multiline=False,write_tab=False)
		self.creatinine.focus=True
		self.add_widget(self.creatinine)
		
		self.add_widget(Label(text="Sex"))
		self.sex = TextInput(multiline=False,write_tab=False)
		self.sex.focus=True
		self.add_widget(self.sex)

		self.add_widget(Label(text="Ejection Fraction"))
		self.EjectionFraction = TextInput(multiline=False,write_tab=False)
		self.EjectionFraction.focus=True
		self.add_widget(self.EjectionFraction)

		self.add_widget(Label(text="High Blood Pressure"))
		self.HighBloodPressure = TextInput(multiline=False,write_tab=False)
		self.HighBloodPressure.focus=True
		self.add_widget(self.HighBloodPressure)

		self.add_widget(Label(text="Platelets"))
		self.Plateles = TextInput(multiline=False,write_tab=False)
		self.Plateles.focus=True
		self.add_widget(self.Plateles)

		self.add_widget(Label(text="Serum Creatinine"))
		self.SerumCreatinine = TextInput(multiline=False,write_tab=False)
		self.SerumCreatinine.focus=True
		self.add_widget(self.SerumCreatinine)

		self.add_widget(Label(text="Serum Sodium"))
		self.SerumSodium = TextInput(multiline=False,write_tab=False)
		self.SerumSodium.focus=True
		self.add_widget(self.SerumSodium)

		self.add_widget(Label(text="Smoking"))
		self.Smoking = TextInput(multiline=False,write_tab=False)
		self.Smoking.focus=True
		self.add_widget(self.Smoking)

		self.add_widget(Label(text="Smokes/day"))
		self.Smokesperday = TextInput(multiline=False,write_tab=False)
		self.Smokesperday.focus=True
		self.add_widget(self.Smokesperday)


		self.connect = Button(text ="Proceed")
		self.connect.bind(on_press=self.connect_btn)
		self.add_widget(self.connect)

		Window.bind(on_key_down=self.on_keyboard_down)


	def on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):
		if keycode == 36:

			self.connect_btn(instance)



	def connect_btn(self,prediction):
		ds = pd.read_csv("C:/Users/IRPT/Desktop/Diabetes App/Diabetes Death Dataset.csv") 
		X = ds.drop(['Diabetes'],axis=1)
		Y = ds['Diabetes']

		sc = StandardScaler()
		sc.fit(X)
		std_data = sc.transform(X)
		X = std_data
		Y = ds['Diabetes']

		X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

		classifier = svm.SVC(kernel='linear')
		classifier.fit(X_train, Y_train)

		X_train_prediction = classifier.predict(X_train)
		training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
		X_test_prediction = classifier.predict(X_test)
		test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
		input_data=tuple([self.age.text,self.anaemia.text,self.creatinine.text,self.sex.text,self.EjectionFraction.text,self.HighBloodPressure.text,self.Plateles.text,self.SerumCreatinine.text,self.SerumSodium.text,self.Smoking.text,self.Smokesperday.text])

		input_data_as_numpy_array = np.asarray(input_data)
		input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
		std_data = sc.transform(input_data_reshaped)
		prediction = classifier.predict(std_data)	

		if (prediction[0]==0):
			self.createPopUp("You do not have diabetes")
		else:
			self.createPopUp("You have Diabetes")
	
	def createPopUp(self,title):

		box = BoxLayout(orientation = 'vertical', padding = (0.1))

		btn1 = Button(text = "Ok")

		box.add_widget(btn1)


		popup = Popup(title=title, title_size= (30),title_align = 'center', content = box,size_hint=(None, None), size=(430, 200), auto_dismiss = True)

		btn1.bind(on_press = popup.dismiss)
		popup.open()



class Proceed(App):
	def build(self):
		self.title = 'DiabetesApp'

		return LoginUI()

if __name__ == "__main__":
	Proceed().run()