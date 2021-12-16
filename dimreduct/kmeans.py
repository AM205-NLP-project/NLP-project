from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit_predict(red_dat)

x_lower, x_upper = -25., 25
y_lower, y_upper = -25., 25
xx,yy = torch.meshgrid(torch.linspace(x_lower, x_upper, 100), torch.linspace(y_lower, y_upper, 100))
Z = kmeans.predict(np.array(torch.vstack([xx.ravel(), yy.ravel()]).T)).reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
plt.scatter(red_dat[:, 0], red_dat[:, 1], color = 'black')