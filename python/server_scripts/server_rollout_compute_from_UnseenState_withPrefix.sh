#$1 --- the firrst argument --- Defines the Name of the Working Directory (Flat/Rubbles_with_OneLargeSlope)
#$2 --- the second argument --- Defines the name of the Folder for Storing Rollouts going to be computed(i.e. RawTrainingSetRollOuts)
#$3 --- the third argument  --- Defines the name of the Folder Stores the Extracted Unseen State (i.e. Unseen State, not with group child folder)
#$4 --- the forth argument  --- Defines the name of Group Prefix of the UnseenState Files that we want to compute from 
#                               (This also define which sub folder we will store in the rollout folder)
#$5 --- the fifth argument  --- Defines the group prefix giving to the rollout computation
#$6 --- the sixth argument  --- Defines the minimum number/index of the unseen state that we want to compute from (i.e. 1)
#$7 --- the seventh argument--- Defines till which maximum unseen state number/index we want to compute (i.e. 5)
#                               if the same as parameter $6, then we only compute for that particular number/index


#longjob -28day -c "nice bash server_rollout_compute_from_UnseenState.sh Rubbles RawRollOuts_fromUnseenState_1 UnseenState_NN_OriginalForm_TrackTraining Group1 Group1 1 1"

#cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/

#homepath=/home/jiayu/Desktop/MLP_DataSet/

homepath=/afs/inf.ed.ac.uk/group/project/mlp_localobj/

cd $homepath
echo "home path: "$homepath

#Get into the working directory
cd $1
workingDir=$PWD
echo "Current Working Directory: "$workingDir

#mkdir for the rollout folder for storing
mkdir $2
cd $2

#Save the file path
filedir=$PWD
echo "Folder to store RollOut files: "$filedir

#Go to the folder where stores the extracted unseenstate, so home foler -> work folder -> extracted unseen state folder -> group fix
cd $homepath && cd $1 && cd $3 && cd $4
echo "Folder to get Extracted Unseen State: "$PWD

sleep 7

#make list of sequence for unseen state numbering
unseenstateNum=$(seq $6 $7)

echo "-------------------------------"

#Start Computation
for i in $unseenstateNum; do

    echo "Compute from Unseen State found at the "$unseenstateNum"-th time"
    #Build UnseenState label
    unseenState_postfix="UnseenState"$unseenstateNum".p"

    #Loop over all the files
    for filename in *.p; do
        if [[ "$filename" == *"$unseenState_postfix"* ]]; then
            echo "Compute from UnseenState: "$filename
            #python3 -W ignore /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
            python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
            -WorkingDirectory $workingDir \
            -RollOutFolderName $2 \
            -Exp_Prefix $5 \
            -InitConditionType fromFile \
            -InitConditionFilePath $(realpath $filename)
        fi
    done
done





