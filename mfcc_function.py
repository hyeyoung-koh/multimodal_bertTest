def mfcc_extract(myfile):
    y, sr = librosa.load(myfile, sr=None)
    mfcc_extracted = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    print('myfile명:', myfile)
    mfcc_extracted = torch.from_numpy(mfcc_extracted).float()
    print('추출한 mfcc:', mfcc_extracted)
    print('mfcc_extracted의 shape:', mfcc_extracted.shape)
    print('mfcc_extracted.size(0):', mfcc_extracted.size(0))
    mfcc_extracted_reshape = mfcc_extracted.view(mfcc_extracted.size(0), -1)
    mfcc_extracted_reshape = mfcc_extracted.view(1, -1)
    print('mfcc_extracted_reshape의 shape:', mfcc_extracted_reshape.shape)
    # mfcc scaling
    mfcc_scale = sklearn.preprocessing.scale(mfcc_extracted_reshape, axis=1)
    print('mfcc_scale:', mfcc_scale)
    print('mfcc_scale의 shape:', mfcc_scale.shape)
    pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    padded_mfcc = pad2d(mfcc_scale, 6000)
    print('padded_mfcc:', padded_mfcc)
    print('padded_mfcc의 shape:', padded_mfcc.shape)
    #print(librosa.display.specshow(padded_mfcc, x_axis='time'))
    print('mfcc_extracted_reshape의 shape:', mfcc_extracted_reshape.shape)

for i in range(1,11):
    mycsv = pd.read_csv('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip_' + str(i) + 'myfinal.csv', encoding='utf-8-sig')
    for j in range(0,len(mycsv)):
        myfile = 'C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip' + str(i) + '_' + str(j) + '_cut.wav'
        mfcc_extract(myfile)