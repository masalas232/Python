# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:21:53 2018

@author: masal
"""
###############################################################################
# Assignment 1: Game
###############################################################################

"""
DocString:
    
    A) Introduction:
    This is the, !! Try Your Luck !!, gameshow. As the title states, all 
    one needs to win is luck. Contestants will face three games of chance: a coin
    toss, pick a color, and a dice throw. After completing each stage succesfully,
    contestants will then have the opportunity to win one of three of prizes by 
    spinning the prize wheel. 
    
    Round 1: Coin toss
    Round 2: Guess the color
    Round 3: Dice Throw
    Prize Round: Spin prize wheel

    B) Known Bugs and/or Errors:
       None

"""

###############################################################################
#                                !! TRY YOUR LUCK !!
###############################################################################

from sys import exit
from random import randint
###################################
#       Start the Game
###################################

def game_start():
    
    print("""\nHello all, and thank you for joining us once again on America's favorite gameshow:

    ____   ______              __  __                    __               __      ____
   / / /  /_  __/______  __    \ \/ /___  __  _______   / /   __  _______/ /__   / / /
  / / /    / / / ___/ / / /     \  / __ \/ / / / ___/  / /   / / / / ___/ //_/  / / / 
 /_/_/    / / / /  / /_/ /      / / /_/ / /_/ / /     / /___/ /_/ / /__/ ,<    /_/_/  
(_|_)    /_/ /_/   \__, /      /_/\____/\__,_/_/     /_____/\__,_/\___/_/|_|  (_|_)   
                  /____/                                                              
        """)

    
    print("\nAs you all know, and incase you don't, my name is Chase and I am your host.")
   
    print("""
 _________________
|.---------------.|
||               ||
|| Hi, I'm Chase ||
|| ~~~~~~~~~~~~~ ||
||_______________||
'-------. .-------'
        | |    _|/
        | |  ."   ".
        | | /(O)-(O)\
       /_)||   /     |
       |_)||  '-     |     
       \_)|\ '.___.' /   |\/|_
        | | \  \_/  /   _|  '/
        |_|\ '.___.'    \ ) /
        \   \_/\__/\__   |==|
         \    \ /\ /\ `\ |  |
          \    \\//     \|  |
           `\   /\   |\ /   |
             |  ||   |\____/
             |__||__|
             
             """)
             
    print("\nChase: Thank you everyone at home for watching us, and to our lovely audience for joinig us.")
    
    print("\nIt is time to welcome todays contestant.")
    
    print("""
              __   
           .-'  '-.       
          /        )                                 
          |  C   o( 
           \       >      
            )  \  /      ..`'
         .-._ / `'      /////     
        / _    \       ( | /
        |/ )    \\     / _,
        / /      |\   / /
       / /       | \ / /
      (  )       /\ ' /
       \ \      (  `-'
        \ \      Y 
        /\ `-.   |
        |(   ^'  |
        \ \\\\  /
         `-    f|
           |   ||
           |   f/
           j   /
           |   )
           |  |
           /  |
           f  |
           \  |
            | |&
           (   `-._,
            -^-----
            
            """)
    
    print("\nChase: What is your name contestant!?")
    global contestant_name
    contestant_name = input('Input your name: \n')
    
    print(f"\nChase: Where are you joining us from today {contestant_name}?")
    global country
    country = input('What country are you from? \n')
    
    print("\nChase: Anyone from back home cheering you on today?")
    global support
    support =  input('Who is cheering you on? (one word: i.e. parents, not my parents).\n')
    
    print(f"""
\nWelcome {contestant_name} from {country}, excited to have you.
Hopefully everyone from {country} and your {support} bring you all the luck you need today!!
Let's begin, but first a short commercial break.
        """)
    
    input('<Press Enter to start the game>\n')
    
    round_1()
    
def round_1():
    
    print("""Chase: Welcome back to:
    ____   ______              __  __                    __               __      ____
   / / /  /_  __/______  __    \ \/ /___  __  _______   / /   __  _______/ /__   / / /
  / / /    / / / ___/ / / /     \  / __ \/ / / / ___/  / /   / / / / ___/ //_/  / / / 
 /_/_/    / / / /  / /_/ /      / / /_/ / /_/ / /     / /___/ /_/ / /__/ ,<    /_/_/  
(_|_)    /_/ /_/   \__, /      /_/\____/\__,_/_/     /_____/\__,_/\___/_/|_|  (_|_)   
                  /____/                                                              
        """)
    
    print(f"Chase: {contestant_name} it's time to test your luck!!")
    
    print(f"""
Chase: Our first round is the Coin Toss. 
Chase: During this round, {contestant_name} will have two attempts to correctly guess which side the coin flip will land on. Here we go!! \n""")
    
    input('<Press Enter to continue>\n')
    
    chances = 2
    
    while chances > 0:
        coin_toss = randint(0, 1)
        print(f"""
Chase: {contestant_name}, what's your guess:
    1. heads
    2. tails
             """)
            
        coin_land = input("< ")
        coin_land = coin_land.lower()
        
        if '1' in coin_land or 'heads' in coin_land:
            if coin_toss == 0:
                print(f"""
Chase: Well done {contestant_name}, amazing job!!! Now on to Round 2.

 ´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶……..
´´´´´´´´´´´´´´´´´´´´¶¶´´´´´´´´´´¶¶……
´´´´´´¶¶¶¶¶´´´´´´´¶¶´´´´´´´´´´´´´´¶¶……….
´´´´´¶´´´´´¶´´´´¶¶´´´´´¶¶´´´´¶¶´´´´´¶¶…………..
´´´´´¶´´´´´¶´´´¶¶´´´´´´¶¶´´´´¶¶´´´´´´´¶¶…..
´´´´´¶´´´´¶´´¶¶´´´´´´´´¶¶´´´´¶¶´´´´´´´´¶¶…..
´´´´´´¶´´´¶´´´¶´´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´¶´´´´´´´´´´´´¶´¶¶´´´´´´´´´´´´´¶¶´´´´´¶¶….
´´¶¶´´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´¶¶´´´´´¶¶….
´¶¶´´´¶¶¶¶¶¶¶¶¶¶¶´´´´¶¶´´´´´´´´¶¶´´´´´´´¶¶…
´¶´´´´´´´´´´´´´´´¶´´´´´¶¶¶¶¶¶¶´´´´´´´´´¶¶….
´¶¶´´´´´´´´´´´´´´¶´´´´´´´´´´´´´´´´´´´´¶¶…..
´´¶´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´¶¶….
´´¶¶´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´´´´´¶¶….
´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´¶¶´´´´´´´´´´´´¶¶…..
´´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶¶¶…….

                      """)
                
                input('<Press Enter to continue>\n')
                round_2()
                break
            
            else:
                chances -= 1
                print(f"""
Chase: OOhh, I'm sorry {contestant_name} that's incorrect.

███████▄▄███████████▄
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓███░░░░░░░░░░░░█
██████▀░░█░░░░██████▀
░░░░░░░░░█░░░░█
░░░░░░░░░░█░░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░░▀▀

Chase: You have {chances} chance left.

                    """)
                
               
                
        elif '2' in coin_land or 'tails' in coin_land:
            if coin_toss == 1:
                print(f"""
Chase: Well done {contestant_name}, amazing job!!! Now on to Round 2.

´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶……..
´´´´´´´´´´´´´´´´´´´´¶¶´´´´´´´´´´¶¶……
´´´´´´¶¶¶¶¶´´´´´´´¶¶´´´´´´´´´´´´´´¶¶……….
´´´´´¶´´´´´¶´´´´¶¶´´´´´¶¶´´´´¶¶´´´´´¶¶…………..
´´´´´¶´´´´´¶´´´¶¶´´´´´´¶¶´´´´¶¶´´´´´´´¶¶…..
´´´´´¶´´´´¶´´¶¶´´´´´´´´¶¶´´´´¶¶´´´´´´´´¶¶…..
´´´´´´¶´´´¶´´´¶´´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´¶´´´´´´´´´´´´¶´¶¶´´´´´´´´´´´´´¶¶´´´´´¶¶….
´´¶¶´´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´¶¶´´´´´¶¶….
´¶¶´´´¶¶¶¶¶¶¶¶¶¶¶´´´´¶¶´´´´´´´´¶¶´´´´´´´¶¶…
´¶´´´´´´´´´´´´´´´¶´´´´´¶¶¶¶¶¶¶´´´´´´´´´¶¶….
´¶¶´´´´´´´´´´´´´´¶´´´´´´´´´´´´´´´´´´´´¶¶…..
´´¶´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´¶¶….
´´¶¶´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´´´´´¶¶….
´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´¶¶´´´´´´´´´´´´¶¶…..
´´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶¶¶…….

                      """)
                
                input('<Press Enter to continue>\n')
                round_2()
                break
            
            else:
                chances -= 1
                print(f"""
Chase: OOhh, I'm sorry {contestant_name} that's incorrect.


███████▄▄███████████▄
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓███░░░░░░░░░░░░█
██████▀░░█░░░░██████▀
░░░░░░░░░█░░░░█
░░░░░░░░░░█░░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░░▀▀

Chase: You have {chances} chance left.

                    """)
        
        else:
            print("Sorry invalid entry, please try again.")
                              
    fail()
    
def round_2():
    
    print(f"""
Chase: Round 2 is guess the color.
Chase: In this round {contestant_name} will try to guess the color of the day. 
Chase: {contestant_name} will have two opportunities to guess the color of the day
chosen at random by our audience members.
Chase: Colors are limited to blue, yellow, or red.\n""")
    
    input('<Press Enter to continue>\n')
    
    chances = 2
    color_of_day = randint(0, 2)
    while chances > 0:
        
        print(f"""
Chase: {contestant_name}, what is your guess?
    1. blue
    2. yellow
    3. red""")
    
        color = input('< ')
        color = color.lower()
        if '1' in color or 'blue' in color:
            if color_of_day == 0:
                print(f"""
Chase: Well done {contestant_name}, amazing job!!! Now on to Round 3.

 ´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶……..
´´´´´´´´´´´´´´´´´´´´¶¶´´´´´´´´´´¶¶……
´´´´´´¶¶¶¶¶´´´´´´´¶¶´´´´´´´´´´´´´´¶¶……….
´´´´´¶´´´´´¶´´´´¶¶´´´´´¶¶´´´´¶¶´´´´´¶¶…………..
´´´´´¶´´´´´¶´´´¶¶´´´´´´¶¶´´´´¶¶´´´´´´´¶¶…..
´´´´´¶´´´´¶´´¶¶´´´´´´´´¶¶´´´´¶¶´´´´´´´´¶¶…..
´´´´´´¶´´´¶´´´¶´´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´¶´´´´´´´´´´´´¶´¶¶´´´´´´´´´´´´´¶¶´´´´´¶¶….
´´¶¶´´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´¶¶´´´´´¶¶….
´¶¶´´´¶¶¶¶¶¶¶¶¶¶¶´´´´¶¶´´´´´´´´¶¶´´´´´´´¶¶…
´¶´´´´´´´´´´´´´´´¶´´´´´¶¶¶¶¶¶¶´´´´´´´´´¶¶….
´¶¶´´´´´´´´´´´´´´¶´´´´´´´´´´´´´´´´´´´´¶¶…..
´´¶´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´¶¶….
´´¶¶´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´´´´´¶¶….
´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´¶¶´´´´´´´´´´´´¶¶…..
´´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶¶¶……. 

                      """)
        
                input('<Press Enter to continue>\n')
                round_3()
                break
            
            else:
                chances -= 1
                print(f"""
Chase: OOhh, I'm sorry {contestant_name} that's incorrect. 


███████▄▄███████████▄
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓███░░░░░░░░░░░░█
██████▀░░█░░░░██████▀
░░░░░░░░░█░░░░█
░░░░░░░░░░█░░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░░▀▀

Chase: You have {chances} chance left.

                    """)
                
        elif '2' in color or 'yellow' in color:
             if color_of_day == 1:
                 print(f"""
Chase: Well done {contestant_name}, amazing job!!! Now on to Round 3.

    ´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶……..
´´´´´´´´´´´´´´´´´´´´¶¶´´´´´´´´´´¶¶……
´´´´´´¶¶¶¶¶´´´´´´´¶¶´´´´´´´´´´´´´´¶¶……….
´´´´´¶´´´´´¶´´´´¶¶´´´´´¶¶´´´´¶¶´´´´´¶¶…………..
´´´´´¶´´´´´¶´´´¶¶´´´´´´¶¶´´´´¶¶´´´´´´´¶¶…..
´´´´´¶´´´´¶´´¶¶´´´´´´´´¶¶´´´´¶¶´´´´´´´´¶¶…..
´´´´´´¶´´´¶´´´¶´´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´¶´´´´´´´´´´´´¶´¶¶´´´´´´´´´´´´´¶¶´´´´´¶¶….
´´¶¶´´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´¶¶´´´´´¶¶….
´¶¶´´´¶¶¶¶¶¶¶¶¶¶¶´´´´¶¶´´´´´´´´¶¶´´´´´´´¶¶…
´¶´´´´´´´´´´´´´´´¶´´´´´¶¶¶¶¶¶¶´´´´´´´´´¶¶….
´¶¶´´´´´´´´´´´´´´¶´´´´´´´´´´´´´´´´´´´´¶¶…..
´´¶´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´¶¶….
´´¶¶´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´´´´´¶¶….
´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´¶¶´´´´´´´´´´´´¶¶…..
´´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶¶¶…….

                        """)
                 
                 input('<Press Enter to continue>\n')
                 round_3()
                 break
             
             else:
                 chances -= 1
                 print(f"""
Chase: OOhh, I'm sorry {contestant_name} that's incorrect.


███████▄▄███████████▄
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓███░░░░░░░░░░░░█
██████▀░░█░░░░██████▀
░░░░░░░░░█░░░░█
░░░░░░░░░░█░░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░░▀▀

Chase: You have {chances} chance left.

                        """)
                 
        elif '3' in color or 'red' in color:
            if color_of_day == 2:
                print(f"""
Chase: Well done {contestant_name}, amazing job!!! Now on to Round 3.

  ´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶……..
´´´´´´´´´´´´´´´´´´´´¶¶´´´´´´´´´´¶¶……
´´´´´´¶¶¶¶¶´´´´´´´¶¶´´´´´´´´´´´´´´¶¶……….
´´´´´¶´´´´´¶´´´´¶¶´´´´´¶¶´´´´¶¶´´´´´¶¶…………..
´´´´´¶´´´´´¶´´´¶¶´´´´´´¶¶´´´´¶¶´´´´´´´¶¶…..
´´´´´¶´´´´¶´´¶¶´´´´´´´´¶¶´´´´¶¶´´´´´´´´¶¶…..
´´´´´´¶´´´¶´´´¶´´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´´´´´´¶¶….
´´´¶´´´´´´´´´´´´¶´¶¶´´´´´´´´´´´´´¶¶´´´´´¶¶….
´´¶¶´´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´¶¶´´´´´¶¶….
´¶¶´´´¶¶¶¶¶¶¶¶¶¶¶´´´´¶¶´´´´´´´´¶¶´´´´´´´¶¶…
´¶´´´´´´´´´´´´´´´¶´´´´´¶¶¶¶¶¶¶´´´´´´´´´¶¶….
´¶¶´´´´´´´´´´´´´´¶´´´´´´´´´´´´´´´´´´´´¶¶…..
´´¶´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´´´´´´´´´´´´´´´¶¶….
´´¶¶´´´´´´´´´´´¶´´¶¶´´´´´´´´´´´´´´´´¶¶….
´´´¶¶¶¶¶¶¶¶¶¶¶¶´´´´´¶¶´´´´´´´´´´´´¶¶…..
´´´´´´´´´´´´´´´´´´´´´´´¶¶¶¶¶¶¶¶¶¶¶…….

                       """)
                
                input('<Press Enter to continue>\n')
                round_3()
                break
            
            else:
                chances -= 1
                print(f"""                    
Chase: OOhh, I'm sorry {contestant_name} that's incorrect.


███████▄▄███████████▄
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓███░░░░░░░░░░░░█
██████▀░░█░░░░██████▀
░░░░░░░░░█░░░░█
░░░░░░░░░░█░░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░░▀▀

Chase: You have {chances} chance left.

                    """)

        else:
            print("Sorry invalid entry, please try agin.")               
    fail()
            
def round_3():
    
    print(f"""
Chase: Round 3 is the dice throw.
Chase: In this round {contestant_name} will have two opportunities to roll one 
die and land on a 4 or greater to progress to the prize round.\n""")
    
    input('<Press Enter to roll the die>\n')
    
    chances = 2
    
    while chances > 0:
        
        dice_roll = randint(1, 6)
        print(f"\nDice landed on: {dice_roll}\n")
        
        if dice_roll >= 4:   
            print(f"""

          .''.      .        *''*    :_\/_:     . 
         :_\/_:   _\(/_  .:.*_\/_*   : /\ :  .'.:.'.
     .''.: /\ :   ./)\   ':'* /\ * :  '..'.  -=:o:=-
    :_\/_:'.:::.    ' *''*    * '.\'/.' _\(/_'.':'.'
    : /\ : :::::     *_\/_*     -= o =-  /)\    '  . 
     '..'  ':::'.    * /\ *     .'/.\'.   '  .      .
         .       .    *..* .     .            .      .
          .       .         .   .              .      .
    ____   __________  _   ____________  ___  ___________    ____
   / / /  / ____/ __ \/ | / / ____/ __ \/   |/_  __/ ___/   / / /
  / / /  / /   / / / /  |/ / / __/ /_/ / /| | / /  \__ \   / / / 
 /_/_/  / /___/ /_/ / /|  / /_/ / _, _/ ___ |/ /  ___/ /  /_/_/  
(_|_)   \____/\____/_/ |_/\____/_/ |_/_/  |_/_/  /____/  (_|_) 

Chase: {contestant_name},you have proved luck is on your side today!! We can now begin the prize round!! 
                """)
            
            input('<Press Enter to continue>\n')
            prize_round()
            break
        
        else:
            chances -= 1
            if chances == 0:
                fail()            
            print(f"""
Chase: OOhh, I'm sorry {contestant_name}, not big enough!!


███████▄▄███████████▄
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓█░░░░░░░░░░░░░░█
▓▓▓▓▓▓███░░░░░░░░░░░░█
██████▀░░█░░░░██████▀
░░░░░░░░░█░░░░█
░░░░░░░░░░█░░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░█░░█
░░░░░░░░░░░░▀▀

Chase: You have {chances} chance left.

                    """)
            
            input('<Press Enter to roll the die again>\n')
                        
    fail()
    
def prize_round():
    print(f"""
Chase: Congratulations on making it to the prize round {contestant_name}, luck
is definitely on your side today!!
Chase: All you have to do now is spin the wheel and claim your prize.
Chase: {contestant_name} you have the opportunity to win one of three prizes:
        1. All paid trip to destination of your choice.
        2. All paid staycation in city you currently reside.
        3. One month of any and all bills incurred during month paid. 
Chase: !!Good Luck!!
        """)
    
    input('<Press Enter to roll the prize wheel!!>\n')
    
    wheel_spin = randint(0, 2)
    
    if wheel_spin == 0:
        print(f"""
Chase: Congratulations {contestant_name}, you have won an all expenses paid trip
for you and one other guest. 
Chase: {contestant_name}, How Lucky Are You Today will fly your guest and you 
out to a city of your choosing for 4 days and 3 nights.
Chase: Airfare, a stay in a 4-star hote, and one dinner are included.
Chase: Thank you for playing, How Lucky Are You Today!!""")
        
        input('<Press Enter to exit>')
        exit(0)
        
    elif wheel_spin == 1:
        print(f"""
Chase: Congratulations {contestant_name}, you have won an all expenses paid staycation
in the city in which you currently reside.
Chase: You and a guest of your choosing, will be treated to a 2 night stay in 
a 4-star hotel, a spay day, and two dinner's in the restaurants of your choice.
Chase: Thank you for playing, How Lucky Are You Today!!""")

        input('<Press Enter to exit>')
        exit(0)

    else:
        print(f"""
Chase: Congratulations {contestant_name}, you have won one month of any and 
all bills incurred during the month paid.
Chase: The !! Try Your Luck team will get in contact with you to 
straighten out the details and provide you with your prize.
Chase. Thank you for playing, How Lucky Are You Today!!""") 
        
        input('<Press Enter to exit>')
        exit(0)
        
def fail():
    print(f"""
    __   __  ______  __  __   __    ____  _____ ______   __
   / /   \ \/ / __ \/ / / /  / /   / __ \/ ___// ____/  / /
  / /     \  / / / / / / /  / /   / / / /\__ \/ __/    / / 
 /_/      / / /_/ / /_/ /  / /___/ /_/ /___/ / /___   /_/  
(_)      /_/\____/\____/  /_____/\____//____/_____/  (_)

Chase: OOhh, looks like you lost {contestant_name}!!
Chase: Luck is not on  your side today!!
            """)
    
    print(f"Chase: Would you like to play again {contestant_name}? (Yes/No)")
    replay = input('>: ')
    replay = replay.lower()
    
    if replay == 'yes' or replay == 'y':
        round_1()
        
    else:
        print(f"""

	
         .e$$$$e.
       e$$$$$$$$$$e
      $$$$$$$$$$$$$$
     d$$$$$$$$$$$$$$b
     $$$$$$$$$$$$$$$$
    4$$$$$$$$$$$$$$$$F
    4$$$$$$$$$$$$$$$$F
     $$$" "$$$$" "$$$
     $$F   4$$F   4$$
     '$F   4$$F   4$"
      $$   $$$$   $P
      4$$$$$"^$$$$$%
       $$$$F  4$$$$
        "$$$ee$$$"
        . *$$$$F4
         $     .$
         "$$$$$$"
          ^$$$$
 4$$c       ""       .$$r
 ^$$$b              e$$$"
 d$$$$$e          z$$$$$b
4$$$*$$$$$c    .$$$$$*$$$r
 ""    ^*$$$be$$$*"    ^"
          "$$$$"
        .d$$P$$$b
       d$$P   ^$$$b
   .ed$$$"      "$$$be.
 $$$$$$P          *$$$$$$
4$$$$$P            $$$$$$"
 "*$$$"            ^$$P
    ""              ^"
    
Chase: Thank you for playing {contestant_name}, you're a quitter! Have a nice life!
            """)
        exit(0)
    
###############################################################################
#                              START THE GAME
###############################################################################        
    
game_start()

             
    
    





















































































































