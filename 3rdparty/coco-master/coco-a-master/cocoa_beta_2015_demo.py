
# coding: utf-8

# In[1]:

# general imports
import json
import time
import sys
import random

# drawing imports
import matplotlib.pyplot as plt
import skimage.io as io


# In[2]:

# path variables
# set paths here and you're good to go...

# directory containing coco-a annotations
COCOA_DIR = '/home/amirro/storage/mscoco/'
# coco-a json file
COCOA_ANN = 'cocoa_beta2015.json'
# directory containing VisualVerbnet
VVN_DIR = '/home/amirro/storage/mscoco/'
# vvn json file
VVN_ANN = 'visual_verbnet_beta2015.json'
# directory containing the MS COCO images
COCO_IMG_DIR = '/home/amirro/storage/mscoco/images'
# directory containing the MS COCO Python API
COCO_API_DIR = '/home/amirro/code/3rdparty/coco-master/PythonAPI'
# directory containing the MS COCO annotations
COCO_ANN_DIR = '/home/amirro/storage/mscoco/annotations'


# In[3]:

# load cocoa annotations

print("Loading COCO-a annotations...")
tic = time.time()

with open("{0}/{1}".format(COCOA_DIR,COCOA_ANN)) as f:
    cocoa = json.load(f)

# annotations with agreement of at least 1 mturk annotator
cocoa_1 = cocoa['annotations']['1']
# annotations with agreement of at least 2 mturk annotator
cocoa_2 = cocoa['annotations']['2']
# annotations with agreement of at least 3 mturk annotator
cocoa_3 = cocoa['annotations']['3']

print("Done, (t={0:.2f}s).".format(time.time() - tic))


# In[4]:

# load visual verbnet

print("Loading VisualVerbNet...")
tic = time.time()

with open("{0}/{1}".format(VVN_DIR,VVN_ANN)) as f:
    vvn = json.load(f)

# list of 145 visual actions contained in VVN
visual_actions = vvn['visual_actions']
# list of 17 visual adverbs contained in VVN
visual_adverbs = vvn['visual_adverbs']
    
print("Done, (t={0:.2f}s).".format(time.time() - tic))


# In[5]:

# visual actions in VVN by category

# each visual action is a dictionary with the following properties:
#  - id:            unique id within VVN
#  - name:          name of the visual action
#  - category:      visual category as defined in the paper
#  - definition:    [empty]
#                   an english language description of the visual action
#  - verbnet_class: [empty]
#                   corresponding verbnet (http://verbs.colorado.edu/verb-index/index.php) entry id for each visual action

for cat in set([x['category'] for x in visual_actions]):
    print("Visual Category: [{0}]".format(cat))
    for va in [x for x in visual_actions if x['category']==cat]:
        print("\t - id:[{0}], visual_action:[{1}]".format(va['id'],va['name']))


# In[6]:

# visual adverbs in VVN by category

# each visual adverb is a dictionary with the following properties:
#  - id:            unique id within VVN
#  - name:          name of the visual action
#  - category:      visual category as defined in the paper
#  - definition:    [empty]
#                   an english language description of the visual action

# NOTE: relative_location is the location of the object with respect to the subject.
# It is not with respect to the reference frame of the image.
# i.e. if you where the subject, where is the object with respect to you?

for cat in set([x['category'] for x in visual_adverbs]):
    print("Visual Category: [{0}]".format(cat))
    for va in [x for x in visual_adverbs if x['category']==cat]:
        print("\t - id:[{0}], visual_adverb:[{1}]".format(va['id'],va['name']))


# In[13]:

# each annotation in cocoa is a dictionary with the following properties:

#  - id:             unique id within coco-a
#  - image_id:       unique id of the image from the MS COCO dataset
#  - object_id:      unique id of the object from the MS COCO dataset
#  - subject_id:     unique id of the subject from the MS COCO dataset
#  - visual_actions: list of visual action ids performed by the subject (with the object if present)
#  - visual_adverbs: list of visual adverb ids describing the subject (and object interaction if present)
print("="*30)

# find all interactions between any subject and any object in an image
image_id = 516931
image_interactions = [x for x in cocoa_2 if x['image_id']==image_id]
print(image_interactions)
print("="*30)

# find all interactions of a subject with any object
subject_id = 190190
# NOTE: In this image there is no interaction with guitar cause it is not annotated in MS COCO
subject_interactions = [x for x in cocoa_2 if x['subject_id']==subject_id]
print(subject_interactions)
print("="*30)

# find interactions of all subjects with an object
object_id = 304500
object_interactions = [x for x in cocoa_2 if x['object_id']==object_id]
print(object_interactions)
print("="*30)

# find all interactions containing a certain visual action
va_name = 'play_instrument'
va_id   = [x for x in visual_actions if x['name']==va_name][0]['id']
interactions = [x for x in cocoa_2 if va_id in x['visual_actions']]
print(interactions)
print("="*30)


# In[8]:

# coco-a is organized to be easily integrable with MS COCO

# load coco annotations
ANN_FILE_PATH = "{0}/instances_{1}.json".format(COCO_ANN_DIR,'train2014')

if COCO_API_DIR not in sys.path:
    sys.path.append( COCO_API_DIR )
from pycocotools.coco import COCO

coco = COCO( ANN_FILE_PATH )


# In[9]:

# visualize an image with subject and object
# and print the interaction annotations

# object_id == -1 means that the annotation is describing a subject and not an interaction
interaction  = random.choice([x for x in cocoa_2 if x['object_id']!=-1 if len(x['visual_actions'])>2])
image_id     = interaction['image_id']

subject_id   = interaction['subject_id']
subject_anns = coco.loadAnns(subject_id)[0]

object_id    = interaction['object_id']
object_anns  = coco.loadAnns(object_id)[0]
object_cat   = coco.cats[object_anns['category_id']]['name']

v_actions    = interaction['visual_actions']
v_adverbs    = interaction['visual_adverbs']

print("Image ID:  [{0}]".format(image_id))
print("Subject ID:[{0}]".format(subject_id))
print("Object ID: [{0}], Category: [{1}]".format(object_id,object_cat))

print("\nVisual Actions:")
for va_id in v_actions:
    va = [x for x in visual_actions if x['id']==va_id][0]
    print("  - id:[{0}], name:[{1}]".format(va['id'],va['name']))
    
print("\nVisual Adverbs:")
for va_id in v_adverbs:
    va = [x for x in visual_adverbs if x['id']==va_id][0]
    print("  - id:[{0}], name:[{1}]".format(va['id'],va['name']))

img = coco.loadImgs(image_id)[0]
I = io.imread("{0}/{1}/{2}".format(COCO_IMG_DIR,'train2014',img['file_name']))
plt.figure(figsize=(12,8))
plt.imshow(I)
coco.showAnns([subject_anns,object_anns])
plt.show()
