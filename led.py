#import RPi.GPIO as GPIO
from gpiozero import LED
from gpiozero.pins.pigpio import PiGPIOFactory
import time
import database
from multiprocessing import Process
import sys
factory1 = PiGPIOFactory(host='192.168.0.105') #connect to raspberry pi
#define gpio pin
red1=LED(26,pin_factory=factory1) 
yellow1=LED(19,pin_factory=factory1)
green1=LED(13,pin_factory=factory1)

factory2 = PiGPIOFactory(host='192.168.0.105')
red2=LED(6,pin_factory=factory2)
yellow2=LED(5,pin_factory=factory2)
green2=LED(11,pin_factory=factory2)

factory3 = PiGPIOFactory(host='192.168.0.105')
red3=LED(2,pin_factory=factory3)
yellow3=LED(4,pin_factory=factory3)
green3=LED(3,pin_factory=factory3)

factory4 = PiGPIOFactory(host='192.168.0.105')
red4=LED(14,pin_factory=factory4)
yellow4=LED(15,pin_factory=factory4)
green4=LED(18,pin_factory=factory4)


red1.on()
red2.on()
red3.on()

flag1=0
flag2=1
flag3=0
#
def time_1(j):
    
    if j!=0:
        return j*1
    else:
        return 0  
        
def mutual(depend_v1, depend_v2):
	time_of_signal = depend_v1+(0.70*depend_v2) #collecting the time dependent signal
	
	return time_of_signal         
c=0  
def RTA(VD,RTA):
	i=(1+VD)*RTA
	return i
	
def A_thread(A):
	
	print('turn on A light') 
	
	red1.on()
	
	time.sleep(2)
	
	red1.off() 
	
	yellow1.on()
	
	time.sleep(1)
	
	yellow1.off()
	
	green1.on()
	
	time.sleep(time_1(A))
	
	green1.off()
				
	yellow1.on()
	
	time.sleep(1)
	
	yellow1.off()
			
	time.sleep(1)
	red1.on()

	
	print('A',A)
	print('turn off A light')
	
	
	
def G_thread(E,A):
	i=mutual(E,A)
	print('e',i)
	print('turn on E light')
		
	red2.on()
	
	time.sleep(1)
	
	red2.off() 
	
	yellow2.on()
	
	time.sleep(1)
	
	yellow2.off()
	
	green2.on()
	
	time.sleep(i)
	
	green2.off()
	
	yellow2.on()
	
	time.sleep(1)
	
	yellow2.off()
	
	time.sleep(1)
	red2.on()
	
	
	
	

			
	print('turn off E light')    


def B_thread(B):
	
	print('turn on B light') 
	time.sleep(B)
	print(B)
	print('turn off B light')

def D_thread(D):
	print('turn on D light') 
	time.sleep(D)
	print(D)
	print('turn off D light')  

	  
def switch(A,B,D,E):

	global flag1,flag2,flag3,flag4,c
	
	database.post_data(A,B,D,E)#function call post the data in database
	
	if A==1 and c==0:
		flag2=1
		c=c+1
	if B==1 and c==0:
		flag1=1
		c=c+1
	if D==1 and c==0:
		flag2=1
		c=c+1		
		
	
	
		
	if A>D and A>B and flag2==1:
		#global flag2,flag1
		print('junction1')
		print('turn off B light')
		#off_B()
		flag2=0
		#print('turn on A light')
		
		t2 = Process(target=G_thread,args=[E,A])
		t1 = Process(target=A_thread,args=[A])
		
		t2.start()
		t1.start()
		#on_A()
		t1.join()		
		t2.join()
		#print('turn on G juntion')
		
		#print(mutual(G,A))
		flag4=1
			
		flag1=1
		flag4=1
		if B>D:
           
			flag1=0
			print('junction1-1')
			flag4=1           
			
           
			
           
			flag2=1
           			
           
			print('turn on B light')
				
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
			
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()
			   
			flag2=0
			
			print ('turn off B light')
			
			flag3=1
           
			print('turn on D light')
			
			
		elif D>B:
			flag1=0
           
			
           
			print('turn off A light')
           
			flag3=1




           		
			print('turn on D light')
			red4.on()
				
			time.sleep(2)
				
			red4.off() 
				
			yellow4.on()
				
			time.sleep(1)
				
			yellow4.off()
				
			green4.on()
				
			time.sleep(time_1(D))
							
			green4.off()
							
			yellow4.on()		
			time.sleep(1)
				
			yellow4.off()
						
			time.sleep(1)
			red4.on()
   
			
			flag3=0
			print('turn off D light')
           
			flag2=1
			
			print('turn on B light') 
							
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
						
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


       
	elif A>D and A>B and flag3==1:
		print('junction1-2')
		print('turn off D light')
		#off_D()
		flag3=0
		print('turn on A ligth')
				
		t2 = Process(target=G_thread,args=[E,A])
		t1 = Process(target=A_thread,args=[A])
		
		t2.start()
		t1.start()
		#on_A()
		t1.join()		
		t2.join()

		#on_A()
		flag1=1
		if B>D:
           
			flag1=0
         	  
			print('turn off A light')
         	  
			flag2=1
           
			
           
			print('turn on B light')
			
							
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
						
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()

			   
			flag2=0

           
			
			
			
			print ('turn off B light')
			
			flag3=1
           
			print('turn on D light')
			red4.on()
				
			time.sleep(2)
				
			red4.off() 
				
			yellow4.on()
				
			time.sleep(1)
				
			yellow4.off()
				
			green4.on()
				
			time.sleep(time_1(D))
				
			green4.off()
							
			yellow4.on()
				
			time.sleep(1)
				
			yellow4.off()
						
			time.sleep(1)
			red4.on()
   
		elif D>B:
			flag1=0
           
			#off_A()
           
			print('turn off A light')
           
			flag3=1
			#on_D(time(D))
			print('turn on D light')
			red4.on()
			
			time.sleep(2)
			
			red4.off() 
			
			yellow4.on()
			
			time.sleep(1)
			
			yellow4.off()
			
			green4.on()
			
			time.sleep(time_1(D))
			
			green4.off()
						
			yellow4.on()
			
			time.sleep(1)
			
			yellow4.off()
					
			time.sleep(1)
			red4.on()
   
           
			#off_D()
			flag3=0
			print('turn off D light')
           
			flag2=1
			#on_B(time(B))
			print('turn on B light') 
							
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
						
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()

			   
			flag2=0

            
                
            
	elif B>A and B>D and flag1==1:
		print('junction2-1')
		print('turn off A light')
		#off_A()
		flag1=0
		print('turn on B light')
		
						
		red3.on()
	
		time.sleep(2)
	
		red3.off() 

		yellow3.on()

		time.sleep(1)

		yellow3.off()

		green3.on()

		time.sleep(time_1(B))
					
		yellow3.on()
	
		time.sleep(1)
	
		yellow3.off()
			
		time.sleep(1)
		red3.on()


		green3.off()
			   
			

		
		#on_B()
		flag2=1
		if A>D:
			print('turn off B light')
            
			#off_B()
            
			flag2=0
			
			
			print('trun on A light')
			flag1=1
            
			print('trun off A light')
			#off_A()
			flag1=0
            
			print('turn on D light')
			#on_D(time(D))
			flag3=1
		elif D>A:
			print('turn off B light') 
			flag2=0
			red4.on()

			time.sleep(2)

			red4.off() 

			yellow4.on()

			time.sleep(1)

			yellow4.off()

			green4.on()

			time.sleep(time_1(D))

			green4.off()
						
			yellow4.on()

			time.sleep(1)

			yellow4.off()
					
			time.sleep(1)
			red4.on()
   
			print('turn on D light')
			
			flag3=1
				    
				    
			print('turn off D light')
			#off_D()
			flag3=0
			print('turn on A light') 
			#on_A(time(A))
			flag1=1 
	elif B>A and B>D and flag3==1:
		print('junction2-2')
		print('turn off D light')
			
		#off_D()
		flag3=0
		print('turn on B light')
								
		red3.on()
	
		time.sleep(2)
	
		red3.off() 

		yellow3.on()

		time.sleep(1)

		yellow3.off()

		green3.on()

		time.sleep(time_1(B))

		green3.off()
		
					
		yellow3.on()
	
		time.sleep(1)
	
		yellow3.off()
			
		time.sleep(1)
		red3.on()


		
		flag2=1
		if A>D:
			print('turn off B light')
				    
			
				    
			flag2=0


			
			flag1=1
				 
				 
			t2 = Process(target=G_thread,args=[E,A])
			t1 = Process(target=A_thread,args=[A])
		
			t2.start()
			t1.start()
						
			t1.join()		
			t2.join()	 
					 
				 
				    
			#print('trun off A light')
			#off_A()
			flag1=0
				    
			print('turn on D light')
			red4.on()
			
			time.sleep(2)
			
			red4.off() 
			
			yellow4.on()
			
			time.sleep(1)
			
			yellow4.off()
			
			green4.on()
			
			time.sleep(time_1(D))
			
			green4.off()
						
			yellow4.on()
			
			time.sleep(1)
			
			yellow4.off()
					
			time.sleep(1)
			red4.on()
   
			flag3=1
		elif D>A:
			print('turn off B light') 
			flag2=0
			#off_B()
			print('turn on D light')
			red4.on()
			
			time.sleep(2)
			
			red4.off() 
			
			yellow4.on()
			
			time.sleep(1)
			
			yellow4.off()
			
			green4.on()
			
			time.sleep(time_1(D))
			
			green4.off()
						
			yellow4.on()
			
			time.sleep(1)
			
			yellow4.off()
					
			time.sleep(1)
			red4.on()
   
			#on_D(time(D))
			flag3=1
				    
				    
			print('turn off D light')
			#off_D()
			flag3=0
			print('turn on A light') 
			#on_A(time(A))
			flag1=1 
	elif D>A and D>A and flag1==1:
		print('junction3-1')
		print ('turn off A light')
			
		#off_A()
		flag1=0
		print ('turn on D light')
		red4.on()
		
		time.sleep(2)
		
		red4.off() 
		
		yellow4.on()
		
		time.sleep(1)
		
		yellow4.off()
		
		green4.on()
		
		time.sleep(time_1(D))
		
		green4.off()
					
		yellow4.on()
		
		time.sleep(1)
		
		yellow4.off()
				
		time.sleep(1)
		red4.on()
   
		#on_D()
		flag3=1
		if B>A:
			print('trun off D light')
			#off_D()
			flag3=0
			print('trun on B light')
											
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
			
						
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
									

			
			print('trun off B light')
			flag2=0
		
			print('turn on A light')
			flag1=1
			
			t2 = Process(target=G_thread,args=[E,A])
			t1 = Process(target=A_thread,args=[A])
						
			t2.start()
			t1.start()
			
			t1.join()		
			t2.join()


		elif A>B:
			print('turn off D light') 
			flag3=0
			#off_D()

			flag1=1
			t2 = Process(target=G_thread,args=[E,A])
			t1 = Process(target=A_thread,args=[A])
			
			t2.start()
			t1.start()
			
			t1.join()		
			t2.join()
			
			#print('trun off A light') 
			flag1=0
			
			print('turn on B light')
			
												
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
			
						
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
				      
	elif D>A and D>A and flag2==1:
		print('junction3-2')
		print ('turn off B light')
		#off_B()
		flag2=0
		print ('turn on D light')
		#on_D()
		flag3=1
		if B>A:
			print('trun off D light')
			#off_D()
			flag3=0
			print('trun on B light')
												
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
			
						
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
			
			print('trun off B light')
			flag2=0
			#off_D()
			print('turn on A light')
			flag1=1
			t2 = Process(target=G_thread,args=[E,A])
			t1 = Process(target=A_thread,args=[A])
			
			t2.start()
			t1.start()
			#on_A()
			t1.join()		
			t2.join()

		elif A>B:
			print('turn off D light') 
			flag3=0
			
			#print('turn on A light')
			flag1=1
			
			
			
			t2 = Process(target=G_thread,args=[E,A])
			t1 = Process(target=A_thread,args=[A])
			
			t2.start()
			t1.start()
			#on_A()
			t1.join()		
			t2.join()
			
			
			#print('trun off A light') 
			flag1=0
			
			print('turn on B light')
			
												
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
			
						
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
	elif A==B and A==D or B==A and B==D or D==A and D==B:
		flag1=1
		#on_A(time_s(A))
		print('turn on A light')

		
		flag1=0
		
		flag2=1
		
		     
		print('turn on B light')
		flag2=0
		
		       
		flag3=1
		
		print('turn on D light')
		
		flag3=0



	  
def switch_1(A,B,D,E):
	global flag1,flag2,flag3,flag4,c
	
	if A==1 and c==0:
		flag2=1
		c=c+1
	if B==1 and c==0:
		flag1=1
		c=c+1
	if D==1 and c==0:
		flag2=1
		c=c+1		
		
	
	
		
	if A>D and A>B and flag2==1:
		#global flag2,flag1
		print('junction1')
		print('turn off B light')
		#off_B()
		flag2=0
		#print('turn on A light')
		
		t2 = Process(target=G_thread,args=[E,A])
		t1 = Process(target=A_thread,args=[A])
		
		t2.start()
		t1.start()
		#on_A()
		t1.join()		
		t2.join()
		#print('turn on G juntion')
		
		#print(mutual(G,A))
		flag4=1
			
		flag1=1
		flag4=1
		if B>D:
           
			flag1=0
			print('junction1-1')
			flag4=1           
			
           
			
           
			flag2=1
           			
           
			print('turn on B light')
				
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(B)

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()

			   
			flag2=0
			
			print ('turn off B light')
			
			flag3=1
           
			print('turn on D light')
			
		elif D>B:
			flag1=0
           
			#off_A()
           
			print('turn off A light')
           
			flag3=1




           		#on_D(time(D))
			print('turn on D light')
			time.sleep(D)
			#off_D()
			flag3=0
			print('turn off D light')
           
			flag2=1
			#on_B(time(B))
			print('turn on B light') 
							
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(B)

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


       
	elif A>D and A>B and flag3==1:
		print('junction1-2')
		print('turn off D light')
		#off_D()
		flag3=0
		print('turn on A ligth')
				
		t2 = Process(target=G_thread,args=[E,A])
		t1 = Process(target=A_thread,args=[A])
		
		t2.start()
		t1.start()
		#on_A()
		t1.join()		
		t2.join()

		#on_A()
		flag1=1
		if B>D:
           
			flag1=0
         	  
			print('turn off A light')
         	  
			flag2=1
           
			
           
			print('turn on B light')
			
							
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()

			   
			flag2=0

           
			flag2=0
			
			
			print ('turn off B light')
			
			flag3=1
           
			print('turn on D light')
		elif D>B:
			flag1=0
           
			#off_A()
           
			print('turn off A light')
           
			flag3=1
			#on_D(time(D))
			print('turn on D light')
           
			#off_D()
			flag3=0
			print('turn off D light')
           
			flag2=1
			#on_B(time(B))
			print('turn on B light') 
							
			red3.on()
	
			time.sleep(2)
	
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(B)

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()
   
			flag2=0

            
                
            
	elif B>A and B>D and flag1==1:
		print('junction2-1')
		print('turn off A light')
		#off_A()
		flag1=0
		print('turn on B light')
		
						
		red3.on()
	
		time.sleep(2)
	
		red3.off() 

		yellow3.on()

		time.sleep(1)

		yellow3.off()

		green3.on()

		time.sleep(time_1(B))

		green3.off()
								
		yellow3.on()
	
		time.sleep(1)
	
		yellow3.off()
			
		time.sleep(1)
		red3.on()

			   
			

		
		#on_B()
		flag2=1
		if A>D:
			print('turn off B light')
            
			#off_B()
            
			flag2=0
			
			
			print('trun on A light')
			flag1=1
            
			print('trun off A light')
			#off_A()
			flag1=0
            
			print('turn on D light')
			#on_D(time(D))
			flag3=1
		elif D>A:
			print('turn off B light') 
			flag2=0
			#off_B()
			print('turn on D light')
			#on_D(time(D))
			flag3=1
				    
				    
			print('turn off D light')
			#off_D()
			flag3=0
			print('turn on A light') 
			#on_A(time(A))
			flag1=1 
	elif B>A and B>D and flag3==1:
		print('junction2-2')
		print('turn off D light')
			
		#off_D()
		flag3=0
		print('turn on B light')
								
		red3.on()
	
		time.sleep(2)
	
		red3.off() 

		yellow3.on()

		time.sleep(1)

		yellow3.off()

		green3.on()

		time.sleep(time_1(B))

		green3.off()
							
		yellow3.on()
	
		time.sleep(1)
	
		yellow3.off()
			
		time.sleep(1)
		red3.on()
	

		#on_B()
		flag2=1
		if A>D:
			print('turn off B light')
				    
			# off_B()
				    
			flag2=0


			#on_A(time(A))
			flag1=1
				 
				 
			t2 = Process(target=G_thread,args=[E,A])
			t1 = Process(target=A_thread,args=[A])
		
			t2.start()
			t1.start()
						#on_A()
			t1.join()		
			t2.join()	 
					 
				 
				    
			#print('trun off A light')
			#off_A()
			flag1=0
				    
			print('turn on D light')
			#on_D(time(D))
			flag3=1
		elif D>A:
			print('turn off B light') 
			flag2=0
			#off_B()
			print('turn on D light')
			#on_D(time(D))
			flag3=1
				    
				    
			print('turn off D light')
			#off_D()
			flag3=0
			print('turn on A light') 
			#on_A(time(A))
			flag1=1 
	elif D>A and D>A and flag1==1:
		print('junction3-1')
		print ('turn off A light')
			
		#off_A()
		flag1=0
		print ('turn on D light')
		#on_D()
		flag3=1
		if B>A:
			print('trun off D light')
			#off_D()
			flag3=0
			print('trun on B light')
											
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(time_1(B))

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
									

			#on_B(time(B))
			print('trun off B light')
			flag2=0
			#off_D()
			print('turn on A light')
			flag1=1
			#on_A(time(A))
		elif A>B:
			print('turn off D light') 
			flag3=0
			#off_D()

			flag1=1
			t2 = Process(target=G_thread,args=[E,A])
			t1 = Process(target=A_thread,args=[A])
			
			t2.start()
			t1.start()
			#on_A()
			t1.join()		
			t2.join()
			#on_A(time(A))
			#print('trun off A light') 
			flag1=0
			#off_A()
			print('turn on B light')
			#on_B(time(B))
												
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(B)

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
				      
	elif D>A and D>A and flag2==1:
		print('junction3-2')
		print ('turn off B light')
		#off_B()
		flag2=0
		print ('turn on D light')
		#on_D()
		flag3=1
		if B>A:
			print('trun off D light')
			#off_D()
			flag3=0
			print('trun on B light')
												
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(B)

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
			
			print('trun off B light')
			flag2=0
			
			print('turn on A light')
			flag1=1
			
		elif A>B:
			print('turn off D light') 
			flag3=0
			``
			flag1=1
			#on_A(time(A))
			
			#thread one initialize
			t2 = Process(target=G_thread,args=[E,A])
			#thread two Thread initialize
			t1 = Process(target=A_thread,args=[A])
			
			#thread one start
			t2.start()
			#thread two start
		
			t1.start()
			#thread join
			t1.join()		
			t2.join()
			
			
			 
			flag1=0
			
			
			
												
			red3.on()
		
			time.sleep(2)
		
			red3.off() 

			yellow3.on()

			time.sleep(1)

			yellow3.off()

			green3.on()

			time.sleep(B)

			green3.off()
									
			yellow3.on()
	
			time.sleep(1)
	
			yellow3.off()
			
			time.sleep(1)
			red3.on()


			flag2=1
	elif A==B and A==D or B==A and B==D or D==A and D==B:
		flag1=1
		#on_A(time_s(A))
		print('turn on A light')

		
		flag1=0
		#off_A()
		flag2=1
		#time.sleep(2)
		#on_B(time_s(B))       
		print('turn on B light')
		flag2=0
		#off_B()
		       
		flag3=1
		#on_D(time_s(D))
		#time.sleep(2)
		print('turn on D light')
		#off_D()
		flag3=0


'''                    
if __name__ == '__main__':

   # defualt1()
    
    while True:
        
        print('A:')
        
        A=float(input())
        print('B:')
        B=float(input())
        print('D:')
        D=float(input())
        print('G')
        E=float(input())
        switch(A,B,D,E)
    #GPIO.cleanup()
'''    
    
    

    

