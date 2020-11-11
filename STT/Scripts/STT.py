'''
Last updated: 11/10/2020
by Xinwen
Please also update the instruction txt.

'''

#Include packages. Install jiwer if needed.
import os
import time
import json
import jiwer

#Let user input file names
name_audio_file = input("Enter audio file name with extension:\t")
name_answer = input("Enter answer file name with extension:\t")
name_setup = input("Enter setup file name with extension:\t")

#navigate to test directory
os.chdir('..')
cwd = os.getcwd()
os.chdir(cwd+'\\'+'Test')
cwd = os.getcwd()

#extract result_position, job_name and media_position from .json file
with open(name_setup,'rb') as setup_file:
    setup = json.load(setup_file)
    result_position = setup['OutputBucketName']
    job_name = setup['MedicalTranscriptionJobName']
    media_position = setup['Media']['MediaFileUri']

#Upload file, create STT job
os.system("aws s3 cp " + name_audio_file + " " + media_position + " --acl public-read")
os.system("aws transcribe start-medical-transcription-job --cli-input-json file://" + name_setup )

#Monitoring job status
CompleteFlag = 0
while CompleteFlag <1:
    state = os.popen('aws transcribe get-medical-transcription-job --medical-transcription-job-name ' + job_name)
    state = state.read()
    if state.find('COMPLETED') != -1:
        CompleteFlag = 1
    else:
        time.sleep(5)
        
#When the job is completed, record the completed information. Generate information.json
os.system('aws transcribe get-medical-transcription-job --medical-transcription-job-name ' + job_name + ' > information.json')

#Load the completed information
path_inform = 'information.json'
with open(path_inform,'rb') as inform_file:
    inform = json.load(inform_file)

#Calc completion time
tms = inform['MedicalTranscriptionJob']['CreationTime'][11:19]
tmc = inform['MedicalTranscriptionJob']['CompletionTime'][11:19]
tm = (int(tmc[0:2]) - int(tms[0:2]))*3600 + (int(tmc[3:5]) - int(tms[3:5]))*60 + (int(tmc[6:8]) - int(tms[6:8]))

#Download the result
os.system("aws s3 cp s3://" + result_position + "/medical/" + job_name + ".json" + " " + cwd)

#Generate ReturnedResult.json
result_path = cwd + "\\" + job_name + ".json"
new_name ="ReturnedResult.json"
new_name = os.path.join(os.path.dirname(result_path), new_name)
os.rename(result_path, new_name)

#Output returned text. Generate TranscribeResult.txt
path_ReturnedResult = 'ReturnedResult.json'
with open(path_ReturnedResult) as ReturnedResult_file:
    ReturnedResult = json.load(ReturnedResult_file)
    text = ReturnedResult['results']['transcripts'][0]['transcript']
    
text_file = open("TranscribeResult.txt", "w")
res_write = text_file.write(text)
text_file.close()

#Load data, generate dict
ans_f = open(name_answer, "r", encoding='utf-8-sig')
ans = ans_f.read()
ans = ans.lower()
ans = ans.replace(',', '')
ans = ans.replace('.', '')
res_f = open("TranscribeResult.txt", "r", encoding='utf-8-sig')
res = res_f.read()
res = res.lower()
res = res.replace(',', '')
res = res.replace('.', '')

#Correctness calculation
correctness = jiwer.wer(ans, res)

#Print the result
print("The time used for job completion is " + str(tm) + " seconds, the correctness is " + str(correctness))

#Hold the window
input('Press Enter to Exit...')

