################################# Projet reconnaissance visage : serveur ##########################################
import socket
import os
import time
from ProgrammeDetectionVisage import DetectionVisage

fichier=["Init"]
while (True):
    fichier=os.listdir("/var/www/html/uploads")                     # lecture du ficher dans le dossier uploads 
    if not fichier :
        print ("dossier vide")                                      # si le dossier est vide on ne fait rien     
    else :
        print ("dossier pas vide")
        nom = DetectionVisage("/var/www/html/uploads/"+fichier[0])  # appel de la fonction de reconnaissance de visage avec l'image trouve dans le dossier
        os.remove("/var/www/html/uploads/"+fichier[0])              # suppression de l image dans le dossier
    time.sleep(1)



