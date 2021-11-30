#External Inputs
#$1 --- Working Direcotry
#$2 --- Folder name for Saving the Experiments
#$3 --- Folder name for the one Saves Ground Truth Training Set
#$4 --- Prefix of the Experiment File name (that will be saved)
#$5 --- The path of Machine learning folder (NN_Model_Valid)

#Command Example (On server):
#longjob -28day -c "nice bash server_NN_trackExp_TrainingSet_All.sh Rubbles NN_TrackingExp_LargeTrainingSet CleanTrainingSetRollOuts_Large Group1 /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles/ML_Models/NN_Model_Valid_LargeTrainingSet/"
#---------------------
#Get the path of the Working Directory
cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/
#cd /home/jiayu/Desktop/MLP_DataSet/
#mkdir $1
cd $1
pwd
filedir=$PWD

echo $filedir

compute_rounds=1

#Get the path of Training RollOuts
rootfolder=/afs/inf.ed.ac.uk/group/project/mlp_localobj/ 
#rootfolder=/home/jiayu/Desktop/MLP_DataSet/ 
workingfolder=$1
slash=/
rooloutfolder=$3
cd $rootfolder$workingfolder$slash$rooloutfolder$slash

#Run the Optimization
for i in *.p; do #Random loop control, just loop over all the files
    
    echo "Iteration: " $compute_rounds

   #We use randomly sampled files
    #tempfilename=$(ls -1 *.p | shuf -n 1)
    #envfile=$(realpath $tempfilename)
    #echo $envfile

   #Or we use the first few files from the list
    echo "$i"
    envfile=$(realpath $i)

    #python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \ 
    python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
    -WorkingDirectory $filedir \
    -RollOutFolderName $2 \
    -Exp_Prefix $4 \
    -EnvModelPath $envfile \
    -LocalObjTrackingFlag Yes \
    -LocalObj_from NeuralNetwork \
    -ML_ModelPath $5

     let compute_rounds=compute_rounds+1

   # if  [[ $compute_rounds -gt $4 ]]; then
   #    break
   # fi

done