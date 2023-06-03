# Local-Relational-Networks in tensorflow 2.0

```
import tensorflow as tf

class GeometryPrior(tf.keras.layers.Layer):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * tf.random.uniform((1, k, k, 2)) - 1
        self.l1 = tf.keras.layers.Conv2D(int(multiplier * channels), 1)
        self.l2 = tf.keras.layers.Conv2D(channels, 1)

    def call(self, x):
        x = self.l2(tf.nn.relu(self.l1(self.position)))
        return tf.reshape(x, (1, self.channels, 1, self.k ** 2))


class KeyQueryMap(tf.keras.layers.Layer):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = tf.keras.layers.Conv2D(channels // m, 1)

    def call(self, x):
        return self.l(x)


class AppearanceComposability(tf.keras.layers.Layer):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.padding = padding
        self.stride = stride

    def call(self, x):
        key_map, query_map = x
        k = self.k
        
        key_patches = tf.image.extract_patches(
            images=key_map,
            sizes=[1, k, k, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        key_patches_shape = tf.shape(key_patches)

        query_patches = tf.image.extract_patches(
            images=query_map,
            sizes=[1, k, k, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        query_patches_shape = tf.shape(query_patches)

        key_map_unfold = tf.reshape(key_patches, (key_patches_shape[0], key_patches_shape[-1], -1))
        query_map_unfold = tf.reshape(query_patches, (query_patches_shape[0], query_patches_shape[-1], -1))

        key_map_unfold = tf.reshape(key_map_unfold,
         [key_map.shape[0], key_map.shape[3],-1,key_map_unfold.shape[-2] // key_map.shape[3]])
        
        query_map_unfold = tf.reshape(query_map_unfold,
         [query_map.shape[0], query_map.shape[3],-1,query_map_unfold.shape[-2] // query_map.shape[3]])
 
        return key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1]


def combine_prior(appearance_kernel, geometry_kernel):
    return tf.nn.softmax(appearance_kernel + geometry_kernel, axis=-1)


class LocalRelationalLayer(tf.keras.layers.Layer):
    def __init__(self, channels, k=7, stride=1, m=8, padding=3):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m
        self.padding = padding
        self.kmap = KeyQueryMap(channels, self.m)
        self.qmap = KeyQueryMap(channels, self.m)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, channels // self.m)
        self.final1x1 = tf.keras.layers.Conv2D(channels, 1)

    def call(self, x):
        gpk = self.gp(0)
        km = self.kmap(x)
        qm = self.qmap(x)
        ak = self.ac((km, qm))
        ck = combine_prior(ak, gpk)[:, None, :, :, :]
        
        #unfold
        x_unfold = tf.image.extract_patches(
            images=x,
            sizes=[1, self.k, self.k, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        x_unfold_shape = tf.shape(x_unfold)     
        x_unfold = tf.reshape(x_unfold, (x_unfold_shape[0], x_unfold_shape[-1], -1))
        x_unfold = tf.reshape(x_unfold, (x.shape[0], self.m, x.shape[3] // self.m,-1, x_unfold.shape[-2] // x.shape[3]))
        
        pre_output = ck * x_unfold
        h_out = (x.shape[1] + 2 * self.padding - self.k) // self.stride + 1
        w_out = (x.shape[2] + 2 * self.padding - self.k) // self.stride + 1
        pre_output = tf.reduce_sum(pre_output, axis=-1)
        pre_output = tf.reshape(pre_output, (x.shape[0], h_out, w_out, x.shape[3]))
        
        return self.final1x1(pre_output)


if __name__ == '__main__':
    x = tf.ones((2, 64, 64, 32))
    lrn = LocalRelationalLayer(32, k=7, padding=3)
    o = lrn(x)
    print(o.shape) #(2, 64, 64, 32)
```
