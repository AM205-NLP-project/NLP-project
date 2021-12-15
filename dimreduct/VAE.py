label_ids = None
classes = None
dat = None

LATENT_DIM = 2
NUM_EPOCHS = 160

class VAE(nn.Module):
    
    def __init__(self, x_dim = 768, h_dim1 = 89, z_dim = LATENT_DIM):
        super(VAE, self).__init__()
        
        # encoding layers
        self.fc1 = nn.Linear(in_features= x_dim, out_features=h_dim1) 
        self.fc21 = nn.Linear(in_features=h_dim1, out_features=z_dim) # mu
        self.fc22 = nn.Linear(in_features=h_dim1, out_features=z_dim) # variance
        
        # mu + variance layers initialize a probability distribution 
        
        # decoding layers
        self.fc3 = nn.Linear(in_features=z_dim, out_features=h_dim1) 
        self.out = nn.Linear(in_features=h_dim1, out_features=x_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 768))
        z = self.sample(mu, log_var) # latent space variance units; one for each latent dimension
        return self.decoder(z), mu, log_var # returns reconstructed image, mean, log variance
    
        # encodes x into z 
    def encoder(self, x):
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)
    
        # decodes z into x
    def decoder(self, z):
        x = F.relu(self.fc3(z))
        return self.out(x) 
        
    def sample(self, mu, log_var): 
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        return eps * std + mu


def main(dat, label_ids, classes):
    ""dat: Word embeddings.
      label_ids: The fit transform of labels.
      classes: Number of unique verbs. """

    VAE = VAE()

    BATCH_SIZE = 3
    training_set = np.array(dat)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE)

    optimizer = optim.Adam(VAE.parameters(), lr=.001) # Adam optimizer
    beta = 1

    def compute_loss(recon_x, x, mu, log_var): 
        
        # reconstruction loss between the output and the input - tries to make reconstruction as accurate as possible
        MSE_func = torch.nn.MSELoss(size_average=False, reduce=True)
        recon_error = MSE_func(recon_x, x.view(-1,768).float())
            
        # KL divergence - tries to push distributions as close as possible to unit Gaussian
        KL_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return 0 * recon_error + beta * KL_div


    stored_training_ELBO, stored_training_MSE = [],[]
    print('Training...')
    print('')

    for epoch in range(1, NUM_EPOCHS+1):
        
        #np.random.shuffle(training_set)
        
        training_MSE = 0
        training_ELBO = 0
        
        for i, train_images in enumerate(train_loader):
            optimizer.zero_grad()
            train_images = train_images[0]
            recon_batch, mu, log_var = VAE(train_images.float())
            ELBO = compute_loss(recon_batch, train_images.float(), mu, log_var)
            MSE = ((recon_batch - train_images.view(-1,768).float())**2).mean(-1).sum()
            training_ELBO += ELBO.item()
            training_MSE += MSE.item()
            ELBO.backward() 
            optimizer.step() 
         
        stored_training_ELBO.append(training_ELBO/(len(training_set)*768))
        stored_training_MSE.append(training_MSE/len(training_set))
        
        if epoch == 1 or epoch%5 == 0:

            print('Epoch', epoch, '--', 'Training ELBO:', np.round(training_ELBO/(len(training_set)*768),5), \
                  '--', 'Training MSE:', np.round(training_MSE/len(training_set),5))
                
    print('')
    print('Finished Training')

    latent = VAE.encoder(torch.tensor(dat))
    latent = np.array([l.detach().numpy() for l in latent])[0]

    x = [l[0] for l in latent]
    y = [l[1] for l in latent]

    le = LabelEncoder()
    label_ids = le.fit_transform(label_ids)

    cs = plt.cm.get_cmap('tab10', classes)
    plt.scatter(x, y, c = label_ids, cmap = cs)