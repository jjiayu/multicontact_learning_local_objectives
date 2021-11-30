#$1 --- the firrst argument --- Defines the Name of the Working Directory (Flat/Rubbles_with_OneLargeSlope)
#$2 --- the second argument --- Defines the name of the Folder for Storing Rollouts (i.e. RawTrainingSetRollOuts)
#$3 --- the third argument  --- Defines the prefix for the file names (Group1)
#$4 --- Number of Lookahead --- 2 or something else
#$5 --- Center Z Height of large Slope --- 0.15

#longjob -28day -c "nice bash server_loop_over_large_slope.sh 2stepsdoesnotwork RollOuts_MoveDownSlope 2steps 2 0.0"

cd /afs/inf.ed.ac.uk/group/project/mlp_localobj/
#cd /home/jiayu/Desktop/MLP_DataSet/

mkdir $1
cd $1
pwd
filedir=$PWD

echo $filedir

for i in 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9;
do
   echo "Large Slope Inclination: " $i

   python3 -W ignore /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen_flat_largeslope.py -WorkingDirectory $filedir -RollOutFolderName $2 -Exp_Prefix $3 -NumLookAhead $4 -LargeSlopeAngle $i -LargeSlopeZ $5
   #python3 /home/jiayu/Desktop/multicontact_learning_local_objectives/python/rhp_plan/rhp_gen_flat_largeslope.py -WorkingDirectory $filedir -RollOutFolderName $2 -Exp_Prefix $3 -NumLookAhead $4 -LargeSlopeAngle $i

   
   #if  [[ $compute_rounds -gt $4 ]]; then
   #   break
   #fi
done