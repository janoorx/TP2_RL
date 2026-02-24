from ex_TP2_2 import *
from ex_TP2_1 import *
from ex_TP2_3 import *
from ex_TP3_2 import *
from ex_TP3_1 import *

if __name__ == "__main__":
    while True:
        choice = input("\033[1;36mChoisissez le TP à exécuter (2 ou 3) ou 'q' pour quitter : \033[0;37m")
        if choice == "2":
            choice = input("\033[1;35mChoisissez l'exercice à exécuter (1,2 ou 3) ou 'q' pour quitter : \033[0;37m")
            if choice == "1":
                ex_TP2_1()
            elif choice == "2":
                ex_TP2_2()
            elif choice == "3":
                ex_TP2_3()
            elif choice == "q":
                print("\033[1;32mAu revoir !\033[0;37m")
                break
            else:
                print("\033[1;31mChoix invalide. Veuillez choisir 1, 2 ou 3.\033[0;37m")
        elif choice == "3":
            choice = input("\033[1;35mChoisissez l'exercice à exécuter (1 ou 2) ou 'q' pour quitter : \033[0;37m")
            if choice == "1":
                ex_TP3_1()
            elif choice == "2":
                ex_TP3_2()
            elif choice == "q":
                print("\033[1;32mAu revoir !\033[0;37m")
                break
            else:
                print("\033[1;31mChoix invalide. Veuillez choisir 1 ou 2.\033[0;37m")
        elif choice == "q":
            print("\033[1;32mAu revoir !\033[0;37m")
            break
        else:
            print("\033[1;31mChoix invalide. Veuillez choisir 2 ou 3.\033[0;37m")