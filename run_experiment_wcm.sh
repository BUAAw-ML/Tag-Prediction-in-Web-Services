# Experiment runner script

#EN='02'  #experiment_no

python main.py \
        --experiment_no='tag50_allSample'  \
        --epochs=400 \
        --epoch_step=380 \
        --device_ids=0 \
        --batch-size=8 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTest' \
        --data_path='../datasets/ProgrammerWeb/ProgrammerWeb_tag50_allSample' \
        --use_previousData=0 \
        --model_type='MABert' \
        --method='MultiLabelMAP' \
        --overlength_handle='truncation' \
        --test_description='' \
        --resume=''  \


#方法、epoch_step
# MLPBert, MABert
#batch-size：1，4，8，16
#data_type: All  TrainTest 
#method: MultiLabelMAP semiGAN_MultiLabelMAP
#overlength_handle: truncation  skip


#permute(1,0,2)_rand wcm
#out[:,-1,:]_rand zyz
#permute(1,0,2)_randn hhm
