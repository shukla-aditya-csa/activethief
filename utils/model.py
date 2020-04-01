"""
MIT License

Copyright (c) 2019 Soham Pal, Yash Gupta, Aditya Shukla, Aditya Kanade,
Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import tensorflow as tf
import numpy as np
import time, os, shutil, glob, gc, tempfile, sys
from cfg import cfg, config
from dsl.dsl_marker_v2 import DSLMarker, collect_aux_data
from sss.base_sss import SubsetSelectionStrategy
from sss.random_sss import RandomSelectionStrategy
from sss.adversarial_sss import AdversarialSelectionStrategy
from sss.uncertainty_sss import UncertaintySelectionStrategy
from sss.kcenter_sss import KCenterGreedyApproach
from sklearn.metrics import f1_score
from attacks.fast_gradient import fgm
from attacks.deepfool import deepfool
from dsl.base_dsl import one_hot_labels  

def train_model(model, train_dsl, val_dsl, logdir):
    "Trains the model and saves the best model in logdir"
    num_batches_tr  = train_dsl.get_num_batches()
    num_batches_val = val_dsl.get_num_batches()
    
    num_samples_val = val_dsl.get_num_samples()
    
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph( model.get_graph() ) 
    
    orig_var_list = [v for v in tf.global_variables() if not v.name.startswith('copy_model')]
    saver         = tf.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)
    
    with tf.Session(config =config) as sess:
        sess.run(tf.global_variables_initializer())
        
        curr_acc = None
        best_acc = None
        no_improvement = 0
        
        for epoch in range(1, cfg.num_epochs+1):
            epoch_time = time.time()
            t_loss = []
            for b_tr in range(num_batches_tr): 

                trX, trY = train_dsl.load_next_batch(b_tr)
                
                global_step, _, summary_str, loss = sess.run([
                                                 model.global_step,
                                                 model.train_op,
                                                 model.train_summary,
                                                 model.mean_loss
                                              ],
                                              feed_dict={
                                                  model.X: trX,
                                                  model.labels: trY,
                                                  model.dropout_keep_prob: cfg.dropout_keep_prob
                                              })
                t_loss.append(loss)    
                
                train_writer.add_summary(summary_str, global_step)
                train_writer.flush()
        
            if epoch % cfg.evaluate_every == 0:
                
                curr_acc = compute_evaluation_measure(model, sess, val_dsl, model.sum_correct_prediction)
                
                if best_acc is None or curr_acc > best_acc :
                    best_acc = curr_acc
                    save_path = saver.save(sess, logdir + '/model_epoch_%d' % (epoch) )       
                    print "Model saved in path: %s" % save_path
                    print "[BEST]",
                    
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                    if (no_improvement % cfg.early_stop_tolerance) == 0:
                        break
                
                print "Step: {} \tValAccuracy: {} \tTrainLoss: {}" .format(  global_step, curr_acc, np.mean(t_loss) )  

            print "End of epoch {} (took {} minutes)." .format(epoch, round((time.time() - epoch_time)/60, 2) ) 
        
        
def evaluate(model, dsl, logdir, checkpoint=None):
    num_samples = dsl.get_num_samples()
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        if checkpoint is None:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            saver.restore(sess, checkpoint)
            
        accuracy = compute_evaluation_measure(model, sess, dsl, model.sum_correct_prediction)
        print "Accuracy:", accuracy

        
def compute_f1_measure(model, sess, dsl, use_aux=False, average='macro'):
    assert not model.is_multilabel()
    
    total_measure = 0
    num_batches = dsl.get_num_batches()
    num_samples = dsl.get_num_samples()
    num_classes = model.get_num_classes()
    
    preds = []
    trues = []
    
    dsl.reset_batch_counter()
    
    for step in range(num_batches): 
        if not use_aux:
            X, Y = dsl.load_next_batch()
        else:
            X, _, aux = dsl.load_next_batch(return_aux=use_aux)
            Y         = collect_aux_data(aux, 'true_prob' )
            
        pred  = sess.run(model.predictions,
                          feed_dict={
                              model.X: X,
                              model.dropout_keep_prob: 1.0
                          }
                        )
        preds.append(pred)
        trues.append(Y)
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    trues = np.argmax(trues,axis=-1)
    
    return f1_score(y_true=trues, y_pred=preds , average=average)

def compute_evaluation_measure(model, sess, dsl, measure, use_aux=False):
    total_measure = 0
    num_batches = dsl.get_num_batches()
    num_samples = dsl.get_num_samples()
    num_classes = model.get_num_classes()
    
    preds = []
    
    dsl.reset_batch_counter()
    
    for step in range(num_batches): 
        if not use_aux:
            X, Y = dsl.load_next_batch()
        else:
            X, _, aux = dsl.load_next_batch(return_aux=use_aux)
            Y         = collect_aux_data(aux, 'true_prob' )
            
        measure_val, pred  = sess.run([measure, model.predictions],
                             feed_dict={
                                 model.X: X,
                                 model.labels: Y,
                                 model.dropout_keep_prob: 1.0
                             }
                            )
        preds.append(pred)
        
        total_measure += measure_val
    
    if model.is_multilabel():
        return total_measure/float(num_samples * num_classes)
    else:
        return total_measure/float(num_samples)
    
def get_labels(model, sess, dsl):
    num_batches = dsl.get_num_batches()
    preds = []
    dsl.reset_batch_counter()
    
    for step in range(num_batches): 
        X, Y = dsl.load_next_batch()
            
        pred  = sess.run( model.predictions,
                             feed_dict={
                                 model.X: X,
                                 model.labels: Y,
                                 model.dropout_keep_prob: 1.0
                             }
                         )
        preds.append(pred)
        
    preds = np.concatenate(preds)
    return preds


def get_predictions(sess, model, x, one_hot=False, drop_out=1.0, labels=False ):
    Y      = []
    Y_prob = []
    Y_idx  = []
    
    for start in range(0, len(x), model.get_batch_size()):
    
        Y_b, Y_prob_b, Y_idx_b  = sess.run( [model.predictions_one_hot, model.prob, model.predictions], 
                                            feed_dict={ model.X: x[start:start+model.get_batch_size()],
                                                        model.dropout_keep_prob:drop_out 
                                                      } 
                                          )
        
        Y.append( Y_b )
        Y_prob.append( Y_prob_b )
        Y_idx.append( Y_idx_b )
    
    Y      = np.concatenate(Y)
    Y_prob = np.concatenate(Y_prob)
    Y_idx  = np.concatenate(Y_idx)
    
    if one_hot:
        if labels:
            return Y, Y_idx
        else:
            return Y
    else:
        if labels:
            return Y_prob, Y_idx
        else:
            return Y_prob

# For KCenter
def get_initial_centers(sess, noise_train_dsl_marked, copy_model):
    Y_vec_true = []

    noise_train_dsl_marked.reset_batch_counter()
    for b in range(noise_train_dsl_marked.get_num_batches()):
        trX, _ = noise_train_dsl_marked.load_next_batch()
        trY    = get_predictions(sess, copy_model, trX, labels=False )
        Y_vec_true.append(trY)

    Y_vec_true  = np.concatenate(Y_vec_true)
            
    return Y_vec_true

# For KCenter
def true_initial_centers(sess, noise_train_dsl_marked):
    Y_vec_true = []

    noise_train_dsl_marked.reset_batch_counter()
    for b in range(noise_train_dsl_marked.get_num_batches()):
        trX, _, trY_aux  = noise_train_dsl_marked.load_next_batch(return_idx=False, return_aux=True)                        
        trY              = collect_aux_data(trY_aux, 'true_prob')
        Y_vec_true.append(trY)

    Y_vec_true  = np.concatenate(Y_vec_true)
            
    return Y_vec_true

    
# new train iter        
def train_copynet_iter(true_model, copy_model, train_dsl, valid_dsl, test_dsl, logdir_true, logdir_copy):
    """ Trains the copy_model iteratively"""
    budget = cfg.initial_seed+cfg.val_size+cfg.num_iter*cfg.k
    
    print "budget: " , budget
    
    num_batches_tr   = train_dsl.get_num_batches()
    num_batches_test = test_dsl.get_num_batches()
    num_samples_test = test_dsl.get_num_samples()
    
    num_classes      = true_model.get_num_classes()
    
    batch_size       = train_dsl.get_batch_size()
    
    noise_train_dsl                                  = DSLMarker( train_dsl )    
    noise_train_dsl_marked, noise_train_dsl_unmarked = noise_train_dsl.get_split_dsls()
    
    noise_val_dsl            = DSLMarker( valid_dsl )   
    
    # Create validation set of size cfg.val_size
    for i in range(cfg.val_size):
        noise_val_dsl.mark(i)
    
    noise_val_dsl_marked, noise_val_dsl_unmarked = noise_val_dsl.get_split_dsls()
    
    orig_var_list = [v for v in tf.global_variables() if not v.name.startswith('copy_model')]    
    orig_saver    = tf.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)
    saver         = tf.train.Saver(max_to_keep=cfg.num_checkpoints)

    train_writer = tf.summary.FileWriter(logdir_copy)
    train_writer.add_graph( true_model.get_graph() )
    train_writer.add_graph( copy_model.get_graph() )
     
    train_time = time.time()
        
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
        tf.logging.info('Model restored!')
       
        # Mark initial samples 
        for i in range(cfg.initial_seed):
            noise_train_dsl.mark(i)
                    
        val_label_counts = dict(list(enumerate([0] * num_classes)))
        
        # Mark validation set
        for i in range(noise_val_dsl_marked.get_num_batches()):
            trX, _, idx  = noise_val_dsl_marked.load_next_batch(return_idx=True)
            trY          = get_predictions( sess, true_model, trX, one_hot=cfg.copy_one_hot)
            for k in range( len(trY) ):
                noise_val_dsl.update(idx[k], aux_data={ 'true_prob' : trY[k] } )
        
            for class_ in list(np.argmax(trY, -1)):
                val_label_counts[class_] += 1
        
        print "val label class dist: ", val_label_counts
        
        pred_match = []
        
        
        for i in range( noise_train_dsl_marked.get_num_batches() ):        
            trX, _,  idx  = noise_train_dsl_marked.load_next_batch(return_idx=True)
            trY           = get_predictions( sess, true_model, trX, one_hot=cfg.copy_one_hot)                

            for k in range( len(trY) ):
                noise_train_dsl.update(idx[k], aux_data={ 'true_prob' : trY[k] } )
        
        Y_t     = get_labels(true_model, sess, test_dsl)
        
        print "Number of test samples" , len(Y_t)
        
        for it in range(cfg.num_iter+1):
            print "Processing iteration " , it+1
                                       
            label_counts = dict(list(enumerate([0] * num_classes)))
            
            sess.close()
            sess = tf.Session(config=config)            
            sess.run(tf.global_variables_initializer())
            
            print "Test var value (before restore):", sess.run(copy_model.test_var)
            
            orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
            
            print "Test var value (after restore):", sess.run(copy_model.test_var)
            
            saver         = tf.train.Saver(max_to_keep=cfg.num_checkpoints)

            # shutil.rmtree(logdir_copy, ignore_errors=True, onerror=None)
            # IMPORTANT: do not remove the iteration log directories, only files!!
            files = [file_ for file_ in os.listdir(logdir_copy) if os.path.isfile(os.path.join(logdir_copy, file_))]

            for file_ in files:
                os.remove(os.path.join(logdir_copy, file_))

            train_writer = tf.summary.FileWriter(logdir_copy)
            train_writer.add_graph( true_model.get_graph() )
            train_writer.add_graph( copy_model.get_graph() )

            gc.collect()
                        
            print 'true model acc', compute_evaluation_measure(true_model, sess, test_dsl, true_model.sum_correct_prediction)
            print 'copy model acc', compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction)
            
            print 'true model F1', compute_f1_measure(true_model, sess, test_dsl)
            print 'copy model F1', compute_f1_measure(copy_model, sess, test_dsl)
            
            exit      = False
            curr_loss = None
            best_f1  = None
            no_improvement = 0
            
            for epoch in range(cfg.copy_num_epochs):
                t_loss     = []
                epoch_time = time.time()
                
                print "\nProcessing epoch {} of iteration {}" .format( epoch+1, it+1)
                
                noise_train_dsl_marked.reset_batch_counter()
                noise_train_dsl.shuffle_data()
               
                for i in range(noise_train_dsl_marked.get_num_batches()):
                    trX, _, trY_aux  = noise_train_dsl_marked.load_next_batch(return_idx=False, return_aux=True)                        
                    trY              = collect_aux_data(trY_aux, 'true_prob')
                    
                    trYhat, summary_str, loss, _, global_step = sess.run([
                                                      copy_model.prob,
                                                      copy_model.train_summary,
                                                      copy_model.mean_loss,
                                                      copy_model.train_op,
                                                      copy_model.global_step
                                                   ],
                                                   feed_dict={
                                                       copy_model.X: trX,
                                                       copy_model.labels: trY,
                                                       copy_model.dropout_keep_prob: cfg.dropout_keep_prob
                                                   })
                    t_loss += [loss]
                    
                    if epoch == 0:
                        for class_ in list(np.argmax(trY, -1)):
                            label_counts[class_] += 1

                    train_writer.add_summary(summary_str, global_step)
                    train_writer.flush()

                if (epoch+1) % cfg.copy_evaluate_every  == 0:
                    print('Epoch: {} Step: {} \tTrain Loss: {}'.format(epoch+1, global_step, np.mean(t_loss)))

                    curr_acc = compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction)
                    print "Test Accuracy (True Dataset): {}".format(curr_acc) 

                    curr_f1 = compute_f1_measure(copy_model, sess, test_dsl)
                    print "Test F1 (True Dataset): {}".format(curr_f1) 

                    val_acc = compute_evaluation_measure(copy_model, sess, noise_val_dsl_marked, copy_model.sum_correct_prediction, use_aux=True)
                    
                    val_f1 = compute_f1_measure(copy_model, sess, noise_val_dsl_marked, use_aux=True)
                    
                    if best_f1 is None or val_f1 > best_f1 :
                        best_f1 = val_f1
                        save_path = saver.save(sess, logdir_copy + '/model_step_%d' % (global_step ))
                        print "Model saved in path: %s" % save_path
                        print "[BEST]",

                        no_improvement = 0
                    else:
                        no_improvement += 1
                        
                        if (no_improvement % cfg.early_stop_tolerance) == 0:
                            if np.mean(t_loss) > 1.5:
                                no_improvement = 0
                            else:
                                exit = True

                    print "Valid Acc (Thief Dataset): {}".format(val_acc) 
                    print "Valid F1 (Thief Dataset): {}".format(val_f1) 
                    
                print "End of epoch {} (took {} minutes).".format(epoch+1, round((time.time() - epoch_time)/60, 2))
                
                if exit:
                    print "Number of epochs processed: {} in iteration {}" .format( epoch+1, it+1 ) 
                    break
                                
            saver.restore(sess, tf.train.latest_checkpoint(logdir_copy))

            # Log the best model for each iteration
            iter_save_path = os.path.join(logdir_copy, str(it))
            os.makedirs(iter_save_path)
            print 'Making directory:', iter_save_path             
            print 'copy model accuracy: ', compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction)
            
            Y_copy  = get_labels(copy_model, sess, test_dsl)
            
            print "TA count" , np.sum(Y_t == Y_copy)
            print "Test agreement between source and copy model on true test dataset", np.sum(Y_t == Y_copy)/float( len(Y_t))
            
            if it+1 == cfg.num_iter+1:
                break
            
            X     = []
            Y     = []
            Y_idx = []
            idx   = []
            
            noise_train_dsl_unmarked.reset_batch_counter()
            
            
            print noise_train_dsl_unmarked.get_num_batches()
            
            for b in range(noise_train_dsl_unmarked.get_num_batches()):
                trX, _, tr_idx = noise_train_dsl_unmarked.load_next_batch(return_idx=True)
                
                for jj in tr_idx:
                    assert jj not in noise_train_dsl.marked_set, "MASSIVE FAILURE!!"
                
                trY, trY_idx = get_predictions(sess, copy_model, trX, labels=True )

                X.append(trX)
                Y.append(trY)
                Y_idx.append(trY_idx)
                idx.append(tr_idx)
            
            X      = np.concatenate(X)
            Y      = np.concatenate(Y)
            Y_idx  = np.concatenate(Y_idx)
            idx    = np.concatenate(idx)
            
            for jj in idx:
                assert jj not in noise_train_dsl.marked_set, "MASSIVE FAILURE 2!!"
            
            sss_time = time.time()
            # Core Set Construction
            if cfg.sampling_method == 'random':
                sss = RandomSelectionStrategy(cfg.k, Y)
            elif cfg.sampling_method == 'adversarial':
                sss = AdversarialSelectionStrategy(cfg.k, Y, X, sess, copy_model,K=len(Y))
            elif cfg.sampling_method == 'uncertainty':
                sss = UncertaintySelectionStrategy(cfg.k, Y)
            elif cfg.sampling_method == 'kcenter':
                sss = KCenterGreedyApproach(cfg.k, Y, get_initial_centers(sess, noise_train_dsl_marked, copy_model) )
            elif cfg.sampling_method == 'adversarial-kcenter':
                sss = AdversarialSelectionStrategy(budget, Y, X, sess, copy_model, K=len(Y))
                s2 = np.array(sss.get_subset())
                sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, noise_train_dsl_marked, copy_model) )
            else:
                raise Exception("sampling method {} not implemented" .format( cfg.sampling_method ) ) 

            s = sss.get_subset()
            
            if cfg.sampling_method in ['adversarial-kcenter']:
                s = s2[s]
            
            print "{} selection time:{} min" .format( cfg.sampling_method, round((time.time() - sss_time)/60, 2) )

            if cfg.sampling_method != 'kmeans' and cfg.sampling_method != 'kcenter' :
                assert len(s) == cfg.k            
            
            print "len(s):", len(s)
            print "len(unique(s)):", len(np.unique(s))

            # Log the chosen samples for each iteration
            np.save(os.path.join(iter_save_path, "samples_chosen.npy"), s)
            s = np.unique(s)
                        
            pred_true_count = np.zeros( (num_classes,num_classes), dtype=np.int32 )
            
            trX = [X[e] for e in s]
            
            true_trY, true_trY_idx = get_predictions(sess, true_model, trX, one_hot=cfg.copy_one_hot, labels=True)
            
            foobXs = dict()
            foobYs = dict()
            foobYps = dict()
            
            noise_train_dsl_marked.reset_batch_counter()
            for b in range(noise_train_dsl_marked.get_num_batches()):
                foobX, foobY, foobI = noise_train_dsl_marked.load_next_batch(return_idx=True)
                _, foobYp    = get_predictions(sess, true_model, foobX, labels=True)
                
                for idx1, foobIdx in enumerate(foobI):
                    foobXs[foobIdx] = foobX[idx1]
                    foobYps[foobIdx] = foobYp[idx1]
            
            print "Mark count before:", len(noise_train_dsl.marked)
            
            for jj in idx:
                assert jj not in noise_train_dsl.marked_set, "MASSIVE FAILURE 3!!"
                        
            for i,k in enumerate(s):
                noise_train_dsl.mark( idx[k], aux_data = { 'true_prob' : true_trY[i] } )
                pred_true_count[true_trY_idx[i]][Y_idx[k]] +=1
                
            noise_train_dsl_marked.reset_batch_counter()
            not_found_count = 0
            hit_count = 0
            for b in range(noise_train_dsl_marked.get_num_batches()):
                foobX, foobY, foobI = noise_train_dsl_marked.load_next_batch(return_idx=True)
                _, foobYp    = get_predictions(sess, true_model, foobX, labels=True)
                
                for idx1, foobIdx in enumerate(foobI):
                    if foobIdx in foobXs:
                        hit_count += 1
                        assert np.allclose(foobXs[foobIdx], foobX[idx1])
                        assert np.allclose(foobYps[foobIdx], foobYp[idx1])
                        
                        del foobXs[foobIdx]
                    else:
                        not_found_count += 1
            
            print "Mark count after:", len(noise_train_dsl.marked)
            print "Not found count:", not_found_count
            print "Found count:", hit_count
            print "Found unique:", len(foobYs) - len(foobXs)
            print "Did not find unique:", len(foobXs)
                
            print "Prediction agreement between source and copy model on selected subset is {}" .format( np.trace(pred_true_count) )       
            pred_match.append(pred_true_count)                    
            
            print "End of iteration ", it+1
        
        if pred_match:
            pred_match = np.stack(pred_match,axis=0)
            np.save( os.path.join(logdir_copy,'pred_match.npy'), pred_match) 
            
        print "Copynet training completed in {} time" .format( round((time.time() - train_time)/3600, 2)  )
        print "---Copynet trainning completed---"
