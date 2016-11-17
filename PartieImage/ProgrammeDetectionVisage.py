################################# Projet reconnaissance visage : serveur ##########################################

from skimage import io
from skimage.io import imread, imsave
import Image
import sys
import os
from PIL import Image
import numpy as np

# fonction de recadrage d image
def ImageCrooped (im):

    height, width, depth = im.shape         # calcul des marges du recadrage de l image
    x0 = width/2
    y0h = int(height/3 - 0.2*height)
    y0b = int(height/3 + 0.2*height)
    gauche =width
    droite=0
    moyenne=0
    haut=0
    for y in range(0, 15):                  # calcul de la moyenne du seuil du fond de l'image
        for j in range(0, width-1):
            moyenne=moyenne+im[y,j,1]   
    seuil=moyenne/((15)*(width-1))-30
    print(seuil)     
    for i in range(0, height-1):            # recherche du seuil haut 
        profilTest = im[i,x0,1]
        if profilTest <seuil:
            haut = i
            break;   
    for y in range(y0h, y0b):       
        for j in range(0, width-1):
            profilTestg = im[y,j,1]
            profilTestd= im[y,width-1-j,1]
            if profilTestg <seuil :         # recherche du seuil gauche
                if j<gauche:
                   gauche = j               
            if profilTestd <seuil :
                if (width-1-j)>droite:     # recherche du seuil haut 
                    droite = width-1-j
 
    #recadrage de l'image avec 3 proportions hauteur/largeur differentes                                       
    im_cropped =[im[haut:((droite - gauche)*1.55),gauche:droite],im[haut:((droite - gauche)*1.6),gauche:droite],im[haut:((droite - gauche)*1.75),gauche:droite]]
    return(im_cropped)

# fonction de lecture de l image
def read_images (path , sz= None ):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk ( path ):
        for subdirname in sorted(dirnames) :
            subject_path = os.path.join(dirname , subdirname )
            for filename in sorted(os.listdir(subject_path)):
                try :
                    im = Image.open(os.path.join (subject_path , filename ))
                    im = im.convert ("L")
                    # resize to given size (if given )
                    if (sz is not None ):
                        im = im.resize(sz , Image.ANTIALIAS )
                    X.append (np.asarray (im , dtype =np.uint8 ))
                    y.append (c)
                except IOError :
                    print "I/O error ({0}) : {1} ".format(errno , strerror )
                except :
                    print " Unexpected error :", sys.exc_info() [0]
                    raise
            c = c+1
    return [X,y]
    
def asRowMatrix (X):
    if len (X) == 0:
        return np.array([])
    mat = np.empty((0 , X [0].size), dtype=X [0].dtype )
    for row in X:
        mat = np.vstack((mat,np.asarray(row).reshape(1,-1)))
    return mat
    
def asColumnMatrix (X):
    if len (X) == 0:
        return np.array ([])
    mat = np.empty ((X [0].size , 0) , dtype =X [0].dtype )
    for col in X:
        mat = np.hstack (( mat , np.asarray ( col ).reshape( -1 ,1)))
    return mat  
    
def pca(X, y, num_components =0):
    [n,d] = X.shape
    if ( num_components <= 0) or ( num_components > n):
        num_components = n
    mu = X.mean ( axis =0)
    X = X - mu
    if n>d:
        C = np.dot (X.T,X)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
    else :
        C = np.dot (X,X.T)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
        eigenvectors = np.dot (X.T, eigenvectors )
        for i in xrange (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm ( eigenvectors [:,i])
    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np.linalg.svd (X.T, full_matrices = False )
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu]
    
def project (W, X, mu= None ):
    if mu is None :
        return np.dot (X,W)
    return np.dot (X - mu , W)
    
def reconstruct (W, Y, mu= None ):
    if mu is None :
        return np.dot(Y,W.T)
    return np.dot (Y,W.T) + mu
        
def normalize (X, low , high , dtype = None ):
    X = np.asarray (X)
    minX , maxX = np.min (X), np.max (X)
    # normalize to [0...1].
    X = X - float ( minX )
    X = X / float (( maxX - minX ))
    # scale to [ low...high ].
    X = X * (high - low )
    X = X + low
    if dtype is None :
        return np.asarray (X)
    return np.asarray (X, dtype = dtype )
    
def create_font ( fontname ='Tahoma', fontsize =10) :
    return { 'fontname': fontname , 'fontsize': fontsize }

        
class AbstractDistance ( object ):
    
    def __init__(self , name ):
            self._name = name
    def __call__(self ,p,q):
        raise NotImplementedError (" Every AbstractDistance must implement the __call__method.")
    @property
    def name ( self ):
        return self._name
    def __repr__( self ):
        return self._name
        
class EuclideanDistance ( AbstractDistance ): 
    def __init__( self ):
        AbstractDistance.__init__(self ," EuclideanDistance ")
    def __call__(self , p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum (np.power((p-q) ,2)))
    
class CosineDistance ( AbstractDistance ):
    def __init__( self ):
        AbstractDistance.__init__(self ," CosineDistance ")
    def __call__(self , p, q):
        p = np.asarray (p).flatten ()
        q = np.asarray (q).flatten ()
        return -np.dot(p.T,q) / (np.sqrt (np.dot(p,p.T)*np.dot(q,q.T)))
  
class BaseModel ( object ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        self.dist_metric = dist_metric
        self.num_components = 0
        self.projections = []
        self.W = []
        self.mu = []
        if (X is not None ) and (y is not None ):
            self.compute (X,y)
            
    def compute (self , X, y):
        raise NotImplementedError (" Every BaseModel must implement the compute method.")
        
    def predict (self , X,minDist):
       
        minClass = -1
        Q = project ( self.W, X.reshape (1 , -1) , self.mu)
        for i in xrange (len( self.projections )):
            dist = self.dist_metric ( self.projections [i], Q)
            if dist < minDist :
                minDist = dist
                minClass = self.y[i]
        return minClass, minDist
        
class EigenfacesModel ( BaseModel ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        super ( EigenfacesModel , self ).__init__ (X=X,y=y, dist_metric = dist_metric , num_components = num_components )
        
    def compute (self , X, y):
        [D, self.W, self.mu] = pca ( asRowMatrix (X),y, self.num_components )
        # store labels
        self.y = y
        # store projections
        for xi in X:
            self.projections.append ( project ( self.W, xi.reshape (1 , -1) , self.mu))

################# Fonction principale de reconnaissance de visage #######################
def DetectionVisage(imtestInit):
    imtest=io.imread(imtestInit) #recuperation de l image
    sys.path.append ("..")
    [X,y] = read_images ("/mnt/hgfs/data/Base_Visages")

    # perform a full pca
    [D, W, mu] = pca ( asRowMatrix(X), y)

    # 2 images ( note : eigenvectors are stored by column !)
    E = []
    for i in xrange ( min( len (X), 16)):
        e = W[:,i].reshape(X [0].shape)
        E.append( normalize (e ,0 ,255) )
   
    # reconstruction steps
    steps =[i for i in xrange(10 , min (len(X), 320) , 20)]
    E = []
    for i in xrange (min(len(steps),16)):
        numEvs = steps[i]
        P = project(W[:,0: numEvs],X[0].reshape(1,-1),mu)
        R = reconstruct(W[:,0:numEvs],P,mu)
        #reshape and append to plots
        R = R.reshape(X[0].shape)
        E.append( normalize (R ,0 ,255) )
    oldIndice=9999999999999
    imtestCrooped=[imtest,imtest,imtest]
    imtestCrooped = ImageCrooped(imtest)                                # recuperation de l image retaillee 3 fois
    for i in range(0, 3):                                               # comparaison des 3 images avec la base
        io.imsave("26_6_modif.jpg",imtestCrooped[i])
        imtestCrooped[i] = Image.open("26_6_modif.jpg")
        resolution  = (92,112)                                          # retaillage de l image dans une dimension commune
        imtestCrooped[i] = imtestCrooped[i].resize(resolution)
        imtestCrooped[i].save("s01-modif.jpg")                          # changement de format
        imtestCrooped[i] = imtestCrooped[i].convert ("L")
        test = np.asarray (imtestCrooped[i] , dtype =np.uint8 )
        model = EigenfacesModel (X , y)
        NomBdd = ["Alec","Antoine","Camille","Johan","Jordi","Kibam", "Michael", "Pierre", "Thomas"] # noms de la base de donnees
        model = EigenfacesModel (X , y)
        minDist = np.finfo('float').max
        nom, indice =model.predict(test,minDist)                        # comparaison avec la base de donnee
        print(NomBdd[nom])
        print(indice)
        if indice <oldIndice :                                          # recupperation de la plus faible distance entre notre image et la base
            nomold =nom
            oldIndice= indice
    fichier = open("/var/www/html/resultat.txt", "w")               # Ouvre le fichier pour retourner le resultat
    fichier.write(NomBdd[nomold])                                   # Ecris le resultat qui a ete trouve
    fichier.close()                                                 # Ferme le fichier                          
    print (" predicted =",NomBdd[nomold])

