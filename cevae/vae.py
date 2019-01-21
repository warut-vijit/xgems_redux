from edward.models import Bernoulli, Normal
import tensorflow as tf

from cevae.utils import fc_net

def cevae_decoder(z, h, n_output, keep_prob, reuse=False, nh=3, lamba=1e-4):
    with tf.variable_scope("defaultCredit/cevae/decoder"):
        # specific to IHDP
        binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        contfeats = [i for i in range(25) if i not in binfeats]

        activation = tf.nn.elu

        # p(x|z)
        hx = fc_net(z, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
        logits = fc_net(hx, [h], [[len(binfeats), None]], 'px_z_bin', lamba=lamba, activation=activation)
        x1 = Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px_z')

        mu, sigma = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_cont', lamba=lamba,
                           activation=activation)
        x2 = Normal(loc=mu, scale=sigma, name='gaussian_px_z')

        # p(t|z)
        logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
        t = Bernoulli(logits=logits, dtype=tf.float32)

        # p(y|t,z)
        mu2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
        mu2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
        y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

    return tf.concat([x1,x2], 1)

def cevae_encoder(x_ph, h, d, keep_prob, nh=3, lamba=1e-4):
    with tf.variable_scope("defaultCredit/cevae/encoder"):
        activation = tf.nn.elu

        # q(t|x)
        logits_t = fc_net(x_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
        qt = Bernoulli(logits=logits_t, dtype=tf.float32)
        # q(y|x,t)
        hqy = fc_net(x_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
        mu_qy_t0 = fc_net(hqy, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
        mu_qy_t1 = fc_net(hqy, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
        qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
        # q(z|x,t,y)
        inpt2 = tf.concat([x_ph, qy], 1)
        hqz = fc_net(inpt2, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
        muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                   activation=activation)
        muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba,
                                   activation=activation)

        mu = qt * muq_t1 + (1. - qt) * muq_t0
        sigma = qt * sigmaq_t1 + (1. - qt) * sigmaq_t0

    return mu, sigma

def decoder(z, dim_img, n_hidden, reuse=True):
    y = cevae_decoder(z, n_hidden, dim_img, 1.0, reuse=reuse)
    return y

def encoder(x, n_hidden, n_output, keep_prob):
    mu, sigma = cevae_encoder(x, n_hidden, n_output, keep_prob)
    return mu, sigma
