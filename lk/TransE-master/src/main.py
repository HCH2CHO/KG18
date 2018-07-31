from dataset import KnowledgeGraph
from model import TransE
import os
import tensorflow as tf
import argparse
import datetime
import logging

def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--model', type=str, default='TransE')
    parser.add_argument('--data', type=str, default='WN18')
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=4800)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    # parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
    # parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10)
    args = parser.parse_args()
    args_dict = vars(args)

    #获取当前时间，创建保存模型和数据的目录
    time_now=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_path=os.path.join('../runs',time_now+"_"+args_dict['model']+"_"+args_dict["data"])
    os.mkdir(model_path)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filename=os.path.join(model_path,'log.txt'),
                        filemode='w')

    #创建checkpoint和summary目录
    ckpt_path=os.path.join(model_path,'checkpoint')
    summ_path=os.path.join(model_path,'summary')
    os.mkdir(ckpt_path)
    os.mkdir(summ_path)
    #选择数据
    data_path=os.path.join('../data',args.data)
    #保存参数
    file=open(model_path+'/parameter.txt','w',encoding='utf-8')
    for arg in args_dict.keys():
        file.write('%-20s\t%s\n'%(arg,str(args_dict[arg])))
    file.close()

    #数据预处理
    kg = KnowledgeGraph(data_path,logging)
    kge_model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                       score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate,
                       n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator, model=args.model, log=logging)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        logging.info('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        logging.info('-----Initialization accomplished-----\n\n')
        #kge_model.check_norm(session=sess)
        summary_writer = tf.summary.FileWriter(logdir=summ_path, graph=sess.graph)
        for epoch in range(args.max_epoch):
            #logging.info('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            kge_model.launch_training(session=sess, summary_writer=summary_writer, epoch=epoch+1)
            if (epoch+1) % args.eval_freq == 0:
                kge_model.launch_evaluation(session=sess,summary_writer=summary_writer, epoch=epoch+1)
                saver = tf.train.Saver(max_to_keep=5)
                saver.save(sess, ckpt_path+'/model.ckpt', global_step=epoch)


if __name__ == '__main__':
    main()
