#$1 --- the firrst argument --- Defines the Name of the Working Directory (Flat/Rubbles_with_OneLargeSlope)
#$2 --- the second argument --- Defines the name of the Folder for Storing Rollouts (i.e. RawTrainingSetRollOuts)
#$3 --- the third argument  --- Defines the prefix for the file names (Group1)
#$4 --- the fifth argument  --- define the number of round/step we want to compute within a rollout
#$5 --- the sixth argument  --- defines the number of rollouts we want to compute
#$6 --- the seventh argument--- defines the machine learning model we have

#longjob -28day -c "nice bash server_rollout_compute.sh Rubbles_DataGrow_by_Comparing RawTestSetRollOuts_4Steps Group20 4 30 40"

#cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/
cd /home/jiayu/Desktop/MLP_DataSet/

mkdir $1
cd $1
pwd
filedir=$PWD

echo $filedir

compute_rounds=1

while [ $compute_rounds ];
do
   echo "Iteration: " $compute_rounds

   python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py \
   -WorkingDirectory $filedir \
   -RollOutFolderName $2 \
   -Exp_Prefix $3 \
   -NumofRounds $4 \
   -LocalObjTrackingFlag Yes \
   -LocalObj_from NeuralNetwork \
   -ML_ModelPath $6 \
   -VisualizationFlag No

   #python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen.py -WorkingDirectory $filedir -RollOutFolderName $2 -Exp_Prefix $3

   let compute_rounds=compute_rounds+1
   
   if  [[ $compute_rounds -gt $5 ]]; then
      break
   fi
done