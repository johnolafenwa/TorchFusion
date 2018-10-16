import visdom

class VisdomLogger(object):
    def __init__(self,
        server='http://localhost',
        endpoint='events',
        port=8097,
        ipv6=True,
        http_proxy_host=None,
        http_proxy_port=None,
        env='main',
        send=True,
        raise_exceptions=None,
        use_incoming_socket=True):

        """

        :param server:
        :param endpoint:
        :param port:
        :param ipv6:
        :param http_proxy_host:
        :param http_proxy_port:
        :param env:
        :param send:
        :param raise_exceptions:
        :param use_incoming_socket:
        """

        self.viz = visdom.Visdom(server=server,endpoint=endpoint,port=port,ipv6=ipv6,http_proxy_host=http_proxy_host,http_proxy_port=http_proxy_port,
                                 env=env,send=send,raise_exceptions=raise_exceptions,use_incoming_socket=use_incoming_socket)
        self.env = env
    def log_text(self,text,win,title=None, env=None, opts=None):
        """

        :param text:
        :param win:
        :param title:
        :param env:
        :param opts:
        :return:
        """
        append = self.viz.win_exists(win,env)

        opts = {"title":title} if opts == None else opts

        self.viz.text(text,win,env=env,opts=opts,append=append)
    def log_image(self,image,win,title=None,env=None,opts=None):

        """

        :param image:
        :param win:
        :param title:
        :param env:
        :param opts:
        :return:
        """

        opts = {"title": title} if opts == None else opts

        self.viz.image(image, win, env=env, opts=opts)

    def log_images(self,image,win,title=None,env=None,opts=None,ncol=8,padding=2):

        """

        :param image:
        :param win:
        :param title:
        :param env:
        :param opts:
        :param ncol:
        :param padding:
        :return:
        """

        opts = {"title": title} if opts == None else opts

        self.viz.images(image, win=win, env=env, opts=opts,nrow=ncol,padding=padding)

    def plot_line(self,X,Y,win,title=None, env=None, opts=None,update_mode=None,name=None):

        """

        :param X:
        :param Y:
        :param win:
        :param title:
        :param env:
        :param opts:
        :param update_mode:
        :param name:
        :return:
        """

        opts = {"title":title} if opts == None else opts

        self.viz.line(X=X,Y=Y,win=win,env=env,opts=opts,update=update_mode,name=name)

    def plot_bar(self,X,Y,win,title=None, env=None, opts=None):

        """

        :param X:
        :param Y:
        :param win:
        :param title:
        :param env:
        :param opts:
        :return:
        """

        opts = {"title":title} if opts == None else opts

        self.viz.bar(X=X,Y=Y,win=win,env=env,opts=opts)

    def plot_quiver(self,X,Y,win,title=None,gridX=None,gridY=None, env=None, opts=None):

        """

        :param X:
        :param Y:
        :param win:
        :param title:
        :param gridX:
        :param gridY:
        :param env:
        :param opts:
        :return:
        """

        opts = {"title":title} if opts == None else opts

        self.viz.quiver(X=X,Y=Y,gridX=gridX,gridY=gridY,win=win,env=env,opts=opts)

    def plot_pie(self, X,win, title=None, env=None, opts=None):
        """

        :param X:
        :param win:
        :param title:
        :param env:
        :param opts:
        :return:
        """
        opts = {"title": title} if opts == None else opts

        self.viz.pie(X=X, win=win, env=env, opts=opts)

    def plot_histogram(self, X,win, title=None, env=None, opts=None):
        """

        :param X:
        :param win:
        :param title:
        :param env:
        :param opts:
        :return:
        """
        opts = {"title": title} if opts == None else opts

        self.viz.histogram(X=X, win=win, env=env, opts=opts)

    def plot_boxplot(self, X,win, title=None, env=None, opts=None):
        """

        :param X:
        :param win:
        :param title:
        :param env:
        :param opts:
        :return:
        """
        opts = {"title": title} if opts == None else opts

        self.viz.boxplot(X=X, win=win, env=env, opts=opts)

    def save(self,envs=["main"]):
        """

        :param envs:
        :return:
        """
        self.viz.save(envs)



