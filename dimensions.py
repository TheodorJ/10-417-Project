def conv_dimensions(c_in, h_in, w_in, c_out, stride, pad, k_height, k_width):

	c_out = c_out
	h_out = (h_in - k_height + 2*pad)//stride + 1 
	w_out = (w_in - k_width  + 2*pad)//stride + 1

	return c_out, h_out, w_out

def pool_dimensions(c_in, h_in, w_in, p_size):

	c_out = c_in
	h_out = h_in//p_size
	w_out = w_in//p_size
	
	return c_out, h_out, w_out

