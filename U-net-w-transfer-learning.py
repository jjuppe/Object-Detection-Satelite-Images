
# coding: utf-8

# # Overview
# In this notebook I will use a U-net like architecture for an instance segmentation problem. The problem is to detect boats in satelite imagery. 

# ## Model Parameters
# We might want to adjust these later (or do some hyperparameter optimizations)

# In[258]:


BATCH_SIZE = 2
NB_EPOCHS = 2


# In[260]:


import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util.montage import montage2d as montage
import gc
from skimage.morphology import label
gc.enable()  # memory is tight


ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')


def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


# In[261]:


masks = pd.read_csv(os.path.join('../input',
                                 'train_ship_segmentations.csv'))
print(masks.shape[0], 'masks found')
print('Total number of train images: {}'.format(masks['ImageId'].value_counts().shape[0]))
masks.head()


# # Make sure encode/decode works
# Given the process
# $$  RLE_0 \stackrel{Decode}{\longrightarrow} \textrm{Image}_0 \stackrel{Encode}{\longrightarrow} RLE_1 \stackrel{Decode}{\longrightarrow} \textrm{Image}_1 $$
# We want to check if/that
# $ \textrm{Image}_0 \stackrel{?}{=} \textrm{Image}_1 $
# We could check the RLEs as well but that is more tedious. Also depending on how the objects have been labeled we might have different counts.
# 
# 

# In[262]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
img_0 = masks_as_image(rle_0)
ax1.imshow(img_0[:, :, 0])
ax1.set_title('Image$_0$')
rle_1 = multi_rle_encode(img_0)
img_1 = masks_as_image(rle_1)
ax2.imshow(img_1[:, :, 0])
ax2.set_title('Image$_1$')
print('Check Decoding->Encoding',
      'RLE_0:', len(rle_0), '->',
      'RLE_1:', len(rle_1))


# # Split into training and validation groups
# We stratify by the number of boats appearing so we have nice balances in each set

# In[263]:


masks['numberShips'] = masks.EncodedPixels.map(lambda row: 1 if isinstance(row, str) else 0)

unique_img_ids = masks.groupby('ImageId').agg({'numberShips': 'sum'}).reset_index()
unique_img_ids['file_size_kb'] = (unique_img_ids['ImageId'].
                                  map(lambda c_img_id: 
                                os.stat(os.path.join(train_image_dir,
                                c_img_id)).st_size / 1024))
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]  # keep only 50kb files
print(unique_img_ids.shape[0])
unique_img_ids.head()


# In[264]:


masks.drop('numberShips', axis=1, inplace=True)
masks.head()


# ## Build train and validation set according to number of boats in datasets

# In[265]:


from sklearn.model_selection import train_test_split
train_ids, valid_ids = train_test_split(unique_img_ids, train_size=0.7,
                 test_size = 0.3, 
                 stratify = unique_img_ids['numberShips'],
                                        )
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')


# ### Examine Number of Ship Images
# Here we examine how often ships appear and replace the ones without any ships with 0

# In[267]:


train_df['numberShips'].hist()
plt.show()
valid_df['numberShips'].hist()
train_df.head()
print('maximum number of ships: {}'.format(train_df.numberShips.max()))


# # Undersample Empty Images
# Here we undersample the empty images to get a better balanced group with more ships to try and segment

# In[238]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(ratio={0: 1000, 1: 1200, 2: 1200,
                                3: 1200, 4: 1200, 5: 1200,
                                6: 1200, 7: 1000, 8: 1100,
                                9: 1000, 10: 700, 11: 700,
                                12: 600, 13: 500, 14: 500,
                                15: 450})


# In[272]:


x = np.array(train_df.ImageId).astype(np.str_).reshape(-1, 1)
y = np.array(train_df.numberShips).astype(np.uint8)

x_train, y_train = rus.fit_sample(x, y)


# In[273]:


x_train = pd.DataFrame(x_train).rename({0: 'ImageId'}, axis=1)
train_masks = pd.merge(x_train, train_df)
x_train = pd.merge(x_train, unique_img_ids, how='left', on='ImageId')
x_train.numberShips.hist(bins=15)
train_masks = train_masks.drop(['numberShips', 'file_size_kb'], axis='columns')
train_masks.set_index('ImageId')
train_masks.head()


# # Decode all the RLEs into Images
# 

# In[256]:


def make_image_gen(in_df):
    all_batches = list(in_df.groupby('ImageId'))
    masks = {'ImageId': list(), 'ImageRGB': list(), 'Mask': list()}
    print(all_batches)
    np.random.shuffle(all_batches)
    for c_img_id, c_masks in all_batches:
        print(c_masks)
        rgb_path = os.path.join(train_image_dir, c_img_id)
        c_img = imread(rgb_path)
        c_mask = masks_as_image(c_masks['EncodedPixels'].values)
        masks['ImageId'].append(c_img_id)
        masks['ImageRGB'].append(c_img)
        masks['Mask'].append(c_mask)
        
    return pd.DataFrame.from_dict(masks)


# In[257]:


train = make_image_gen(train_masks)
x_train = train['ImageRGB'].values
y_train = train['Mask'].values




# # Make the Validation Set

# In[18]:


valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))
print(valid_x.shape, valid_y.shape)


# # Augment Data

# In[19]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last',
                  preprocessing_function=preprocess_input)
# brightness can be problematic since it seems to change the labels differently from the images 
if AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
dg_args.pop('preprocessing_function')
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(in_x, 
                             batch_size = BATCH_SIZE, 
                             seed = seed, 
                             shuffle=True)
        g_y = label_gen.flow(in_y, 
                             batch_size = BATCH_SIZE, 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x), next(g_y)


# In[20]:


cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
# only keep first 9 samples to examine in detail


# In[21]:


gc.collect()


# # Build a Model
# Here we use a slight deviation on the U-Net standard

# In[22]:


from keras.applications.vgg16 import VGG16 as VGG16, preprocess_input
encode_model = VGG16(input_shape=(768,768,3), include_top=False, weights='imagenet')
encode_model.trainable = False


# In[23]:


from keras import models, layers

# output and start upsampling
features = encode_model.output

conv_1 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(features)
up_conv = layers.Conv2DTranspose(256, (3,3), strides=(2,2), activation='relu', padding='same')(conv_1)

# first concatenation block
concat_1 = layers.concatenate([encode_model.get_layer('block5_conv3').output, up_conv], axis=-1, name='concat_1')
conv_2 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(concat_1)
up_conv_2 = layers.Conv2DTranspose(256, (3,3), strides=(2,2), activation='relu', padding='same')(conv_2)


# second concatenation block
concat_2 = layers.concatenate([up_conv_2, encode_model.get_layer('block4_conv3').output])
conv_3 = layers.Conv2D(512, (3,3), activation='relu', padding='same')(concat_2)
up_conv_3 = layers.Conv2DTranspose(128, (3,3), strides=(2,2), activation='relu', padding='same')(conv_3)

# third concatenation block
concat_3 = layers.concatenate([up_conv_3, encode_model.get_layer('block3_conv3').output])
conv_4 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(concat_3)
up_conv_4 = layers.Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same')(conv_4)

# fourth concatenation block
concat_4 = layers.concatenate([up_conv_4, encode_model.get_layer('block2_conv2').output])
conv_5 = layers.Conv2D(128, (3,3), activation='relu', padding='same',name='block2_conv')(concat_4)
up_conv_5 = layers.Conv2DTranspose(32, (3,3), strides=(2,2), activation='relu', padding='same')(conv_5)

# fifth concatenation block
concat_4 = layers.concatenate([up_conv_5, encode_model.get_layer('block1_conv2').output])
conv_6 = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(concat_4)

for layer in encode_model.layers:
    layer.trainable = False
    
final_model = models.Model(inputs=[encode_model.input], outputs=[conv_6])
final_model.summary()


# In[24]:


import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


# In[25]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

tensorboard = TensorBoard("./logs", write_images= True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat, tensorboard]


# In[26]:


import tensorflow as tf

def fit():
    final_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
    step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0]//BATCH_SIZE)
    aug_gen = create_aug_gen(make_image_gen(balanced_train_df))
    
    with tf.device("/gpu:0"):
        loss_history = [final_model.fit_generator(aug_gen, 
                             epochs=NB_EPOCHS, 
                            steps_per_epoch=step_count,
                             validation_data=(valid_x, valid_y),
                             callbacks=callbacks_list,
                            workers=1 # the generator is not very thread safe
                                       )]
    return loss_history

counter = 0
while True:
    loss_history = fit()
    counter += 1
    # if np.min([mh.history['val_loss'] for mh in loss_history]) < -0.5 or counter==10:
    if counter == 10:
        break
  


# def show_loss(loss_history):
#     epich = np.cumsum(np.concatenate(
#         [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
#     _ = ax1.plot(epich,
#                  np.concatenate([mh.history['loss'] for mh in loss_history]),
#                  'b-',
#                  epich, np.concatenate(
#             [mh.history['val_loss'] for mh in loss_history]), 'r-')
#     ax1.legend(['Training', 'Validation'])
#     ax1.set_title('Loss')
# 
#     _ = ax2.plot(epich, np.concatenate(
#         [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
#                      epich, np.concatenate(
#             [mh.history['val_true_positive_rate'] for mh in loss_history]),
#                      'r-')
#     ax2.legend(['Training', 'Validation'])
#     ax2.set_title('True Positive Rate\n(Positive Accuracy)')
#     
#     _ = ax3.plot(epich, np.concatenate(
#         [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
#                      epich, np.concatenate(
#             [mh.history['val_binary_accuracy'] for mh in loss_history]),
#                      'r-')
#     ax3.legend(['Training', 'Validation'])
#     ax3.set_title('Binary Accuracy (%)')
#     
#     _ = ax4.plot(epich, np.concatenate(
#         [mh.history['dice_coef'] for mh in loss_history]), 'b-',
#                      epich, np.concatenate(
#             [mh.history['val_dice_coef'] for mh in loss_history]),
#                      'r-')
#     ax4.legend(['Training', 'Validation'])
#     ax4.set_title('DICE')
# 
# show_loss(loss_history)

# In[27]:


final_model.load_weights("models/seg_model_weights.best.hdf5")
final_model.save('segmentation_model.h5')


# pred_y = seg_model.predict(valid_x)
# print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())

# fig, ax = plt.subplots(1, 1, figsize = (10, 10))
# ax.hist(pred_y.ravel(), np.linspace(0, 1, 10))
# ax.set_xlim(0, 1)
# ax.set_yscale('log', nonposy='clip')

# # Prepare Full Resolution Model
# Here we account for the scaling so everything can happen in the model itself

# if IMG_SCALING is not None:
#     fullres_model = models.Sequential()
#     fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
#     fullres_model.add(seg_model)
#     fullres_model.add(layers.UpSampling2D(IMG_SCALING))
# else:
#     fullres_model = seg_model
# fullres_model.save('fullres_model.h5')

# # Run the test data

# test_paths = os.listdir(test_image_dir)
# print(len(test_paths), 'test images found')

# fig, m_axs = plt.subplots(20, 2, figsize = (10, 40))
# [c_ax.axis('off') for c_ax in m_axs.flatten()]
# for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
#     c_path = os.path.join(test_image_dir, c_img_name)
#     c_img = imread(c_path)
#     first_img = np.expand_dims(c_img, 0)/255.0
#     first_seg = fullres_model.predict(first_img)
#     ax1.imshow(first_img[0])
#     ax1.set_title('Image')
#     ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
#     ax2.set_title('Prediction')
# fig.savefig('test_predictions.png')

# # Submission
# Since gneerating the submission takes a long time and quite a bit of memory we run it in a seperate kernel located at https://www.kaggle.com/kmader/from-trained-u-net-to-submission-part-2 
# That kernel takes the model saved in this kernel and applies it to all the test data
